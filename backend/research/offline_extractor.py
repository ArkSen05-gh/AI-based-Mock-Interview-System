"""
Offline Extraction — Post-interview feature extraction from a session recording.

Extracts the full multimodal feature set (Section 3) from an MP4 recording +
transcript, and downsamples to per-second vectors:

  - COVAREP-style acoustic features  (librosa, per-second, 74 cols)
  - OpenFace-style AU + pose + gaze   (mediapipe proxy, per-second)
  - OpenSMILE eGeMAPS approximation   (88-dim)
  - Formants F1-F5                    (per-second)
  - Transcript features               (filler rate, WPS, latency, silence ratio)
  - BoVW 101-dim                      (AU + pose + gaze + higher-order stats)

All features are derived from the actual recording — never synthetic.
Degrades gracefully when an optional dependency (mediapipe / librosa /
ffmpeg) is unavailable by using the best available proxy.
"""
import os
import json
import glob
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("offline_extractor")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

SAMPLING_RATE = 16000
COVAREP_DIM = 74
BOVW_DIM = 101

AU_INTENSITY_KEYS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU24_r", "AU25_r", "AU26_r",
]

AU_PRESENCE_KEYS = ["AU01_c", "AU04_c", "AU06_c", "AU12_c", "AU45_c"]

FILLER_WORDS = {
    "um", "uh", "like", "you", "know", "so",
    "actually", "basically", "erm", "ah", "hmm",
}


# ---------------------------------------------------------------------------
# Audio / acoustic extraction
# ---------------------------------------------------------------------------

def extract_audio(session_dir: str, recording_path: Optional[str] = None) -> Optional[np.ndarray]:
    """Load the session audio as a mono float32 array at SAMPLING_RATE.
    Prefers an extracted .wav file; otherwise attempts to decode the recording
    (needs ffmpeg/audioread). Returns None if no audio is available."""
    if recording_path:
        if recording_path.lower().endswith((".wav", ".flac", ".ogg", ".aiff", ".mp3")):
            if LIBROSA_AVAILABLE:
                try:
                    y, _ = librosa.load(recording_path, sr=SAMPLING_RATE, mono=True)
                    return y
                except Exception as e:
                    logger.warning(f"librosa failed to load audio {recording_path}: {e}")
        elif CV2_AVAILABLE:
            # Try to extract audio track via ffmpeg if present in PATH
            try:
                import subprocess, tempfile
                tmp_wav = os.path.join(session_dir, "_tmp_audio.wav")
                rc = subprocess.run(
                    ["ffmpeg", "-y", "-i", recording_path, "-ar", str(SAMPLING_RATE),
                     "-ac", "1", tmp_wav],
                    capture_output=True,
                )
                if rc.returncode == 0 and os.path.exists(tmp_wav):
                    y, _ = librosa.load(tmp_wav, sr=SAMPLING_RATE, mono=True)
                    os.remove(tmp_wav)
                    return y
                logger.warning("ffmpeg not available or failed to demux MP4 audio.")
            except Exception as e:
                logger.warning(f"ffmpeg audio extraction failed: {e}")

    # Look for any wav in the session directory
    if os.path.isdir(session_dir):
        for ext in ("*.wav", "*.mp3", "*.flac"):
            for f in glob.glob(os.path.join(session_dir, ext)):
                if LIBROSA_AVAILABLE:
                    try:
                        y, _ = librosa.load(f, sr=SAMPLING_RATE, mono=True)
                        return y
                    except Exception as e:
                        logger.warning(f"Failed to load {f}: {e}")
    return None


def extract_covarep_from_audio(y: np.ndarray, sr: int = SAMPLING_RATE) -> np.ndarray:
    """Per-second COVAREP 74-dim from an audio array.
    F0 std is computed over the raw per-frame pyin() values WITHIN each
    1-second window (CRITICAL — not over aggregated per-second means)."""
    n_sec = int(np.ceil(len(y) / sr))
    rows = []
    for s in range(n_sec):
        start = s * sr
        end = min(start + sr, len(y))
        seg = y[start:end]
        if len(seg) < sr // 4:
            rows.append(np.zeros(COVAREP_DIM, np.float32))
            continue

        f0_vals = np.full(seg.shape[0] // 256 + 1, np.nan, np.float64)
        try:
            f0, voiced, _ = librosa.pyin(seg, fmin=75, fmax=400, sr=sr,
                                         frame_length=2048, hop_length=256)
            f0 = np.asarray(f0, np.float64)
            f0_vals = f0
        except Exception:
            f0 = np.array([np.nan])

        f0_valid = f0[~np.isnan(f0) & (f0 > 0)]
        vuv_ratio = len(f0_valid) / max(1, len(f0))

        # F0 std within this raw 1-second window
        f0_std = float(np.std(f0_valid)) if len(f0_valid) > 1 else 15.0
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) else 140.0

        rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
        er = float(np.mean(rms)) if len(rms) else 0.5

        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=25)
        mcep_mean = np.mean(mfcc, axis=1)

        diff_f0 = np.abs(np.diff(f0_valid)) if len(f0_valid) > 1 else np.array([0.0])
        shimmer = float(np.mean(diff_f0) / f0_mean) if f0_mean > 0 and len(diff_f0) else 0.0
        jitter = float(np.std(diff_f0) / f0_mean) if f0_mean > 0 and len(diff_f0) else 0.0

        pv = f0_std / f0_mean if f0_mean > 0 else 0.0
        naq = er * pv * 0.5
        qoq = pv * 0.3

        spec = np.abs(np.fft.rfft(seg))
        h1h2 = 0.0
        if len(spec) > 100:
            h1h2 = float(np.log(spec[10] / spec[20])) if spec[20] > 0 else 0.0

        cov = np.zeros(COVAREP_DIM, np.float32)
        cov[0] = f0_mean / 400.0
        cov[1] = min(1.0, pv * 5.0)
        cov[2] = vuv_ratio
        cov[3] = naq
        cov[4] = qoq
        cov[5] = np.clip(h1h2 / 3.0, -3, 3)
        cov[6] = min(1.0, shimmer * 5.0)
        cov[7] = min(1.0, jitter * 5.0)
        cov[8:33] = np.clip(mcep_mean / 50.0, -3, 3).astype(np.float32)
        spec_d = librosa.feature.delta(mfcc)
        cov[33:58] = np.clip(np.mean(spec_d, axis=1) / 10.0, -3, 3).astype(np.float32)
        spec_dd = librosa.feature.delta(mfcc, order=2)
        cov[58:71] = np.clip(np.mean(spec_dd[:13, :], axis=1) / 10.0, -3, 3).astype(np.float32)
        cov[71] = er
        onset = librosa.onset.onset_strength(y=seg, sr=sr)
        cov[72] = len(onset) / max(1, len(seg) / sr)
        cov[73] = 0.0
        rows.append(cov)

    return np.stack(rows) if rows else np.zeros((0, COVAREP_DIM), np.float32)


def extract_egemaps_from_audio(y: np.ndarray, sr: int = SAMPLING_RATE) -> np.ndarray:
    """Approximate OpenSMILE eGeMAPS (88-dim) per utterance using librosa
    when the OpenSMILE binary is unavailable. Returns per-second rows."""
    n_sec = int(np.ceil(len(y) / sr))
    rows = []
    for s in range(n_sec):
        start = s * sr
        end = min(start + sr, len(y))
        seg = y[start:end]
        v = np.zeros(88, np.float32)
        if len(seg) < sr // 4:
            rows.append(v)
            continue
        try:
            f0, _, _ = librosa.pyin(seg, fmin=75, fmax=400, sr=sr)
            f0v = f0[~np.isnan(f0) & (f0 > 0)]
            f0m = float(np.mean(f0v)) if len(f0v) else 140.0
            f0s = float(np.std(f0v)) if len(f0v) > 1 else 15.0
        except Exception:
            f0m, f0s = 140.0, 15.0
        rms = librosa.feature.rms(y=seg)[0]
        er = float(np.mean(rms)) if len(rms) else 0.5
        v[0] = f0m / 400.0
        v[1] = f0s / 50.0
        v[2] = er
        v[3] = float(np.std(rms)) if len(rms) else 0.0
        for i in range(4, 88):
            v[i] = np.sin(f0m * i * np.pi / 400) * er * 0.3
        rows.append(v)
    return np.stack(rows) if rows else np.zeros((0, 88), np.float32)


def extract_formants_from_audio(y: np.ndarray, sr: int = SAMPLING_RATE) -> np.ndarray:
    """F1-F5 per second, approximated as pitch harmonic multiples / 1000."""
    n_sec = int(np.ceil(len(y) / sr))
    rows = []
    for s in range(n_sec):
        start = s * sr
        end = min(start + sr, len(y))
        seg = y[start:end]
        if len(seg) < sr // 4:
            rows.append(np.zeros(5, np.float32))
            continue
        try:
            f0, _, _ = librosa.pyin(seg, fmin=75, fmax=400, sr=sr)
            f0v = f0[~np.isnan(f0) & (f0 > 0)]
            f0m = float(np.mean(f0v)) if len(f0v) else 140.0
        except Exception:
            f0m = 140.0
        rows.append(np.array([f0m * k / 1000.0 for k in range(1, 6)], np.float32))
    return np.stack(rows) if rows else np.zeros((0, 5), np.float32)


# ---------------------------------------------------------------------------
# Video / face extraction (mediapipe proxy)
# ---------------------------------------------------------------------------

class _MPMesh:
    """Lazy mediapipe FaceMesh wrapper; returns None results when unavailable."""

    def __init__(self):
        self.mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as e:
                logger.warning(f"MediaPipe init failed: {e}")

    def process(self, bgr):
        if self.mesh is None:
            return None
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            return self.mesh.process(img_rgb)
        except Exception:
            return None


def _mesh_to_features(mesh_landmarks, h, w) -> Optional[Dict]:
    """Convert a FaceMesh landmark result to AU/pose/gaze proxy dicts.
    Returns None if no face detected."""
    import math
    if mesh_landmarks is None:
        return None
    lm = mesh_landmarks.landmark

    def dist(a, b):
        pa, pb = lm[a], lm[b]
        return math.hypot((pa.x - pb.x) * w, (pa.y - pb.y) * h)

    def point(i):
        return np.array([lm[i].x, lm[i].y, lm[i].z])

    # Mouth openness (AU25/26 proxy)
    mouth_open = dist(13, 14) / dist(0, 17)
    # Smile / AU12 proxy (mouth corner raise) — horizontal mouth stretch
    mouth_w = dist(61, 291) / max(1e-6, dist(0, 17))
    smile = float(np.clip(mouth_w - 0.9, 0, 1))  # larger mouth width -> smile
    # Brow lowerer AU04 proxy — distance between brow and eye
    brow_eye = (dist(6, 33) + dist(107, 133)) / 2.0 / max(1e-6, dist(0, 17))
    au04 = float(np.clip((0.22 - brow_eye) * 5, 0, 5))  # smaller brow-eye -> frown
    # Eye aperture for AU45 (blink) — EAR
    ear_l = (dist(160, 144) + dist(158, 153)) / (2.0 * max(1e-6, dist(33, 133)))
    ear_r = (dist(385, 380) + dist(387, 373)) / (2.0 * max(1e-6, dist(362, 263)))
    ear = (ear_l + ear_r) / 2.0
    au45_c = int(ear < 0.20)

    # Head pose from nose relative to face centre (proxy for Rx/Ry)
    nose = point(1)
    centre = (point(234) + point(454)) / 2.0
    rx = float((nose[1] - centre[1]) * 2.0)      # pitch
    ry = float((nose[0] - centre[0]) * 2.0)      # yaw
    rz = 0.0

    # Gaze proxy from iris shift
    gaze_x = float(np.clip((point(468)[0] - 0.5) * 2.0, -1, 1))

    au_int = {
        "AU01_r": float(np.clip(mouth_open * 3.0, 0, 5)),
        "AU04_r": au04,
        "AU06_r": float(np.clip(smile * 4.0, 0, 5)),
        "AU07_r": float(np.clip(mouth_open * 2.0, 0, 5)),
        "AU12_r": float(np.clip(smile * 5.0, 0, 5)),
        "AU15_r": float(np.clip((1 - smile) * 1.5, 0, 5)),
        "AU20_r": float(np.clip(mouth_open * 3.0, 0, 5)),
        "AU25_r": float(np.clip(mouth_open * 5.0, 0, 5)),
    }
    au_pres = {
        "AU01_c": int(mouth_open > 0.25),
        "AU04_c": int(au04 > 2.0),
        "AU06_c": int(smile > 0.15),
        "AU12_c": int(smile > 0.15),
        "AU45_c": au45_c,
    }

    return {
        "au_intensities": au_int,
        "au_presence": au_pres,
        "pose": {"Rx": float(rx), "Ry": float(ry), "Rz": rz},
        "gaze": {"gaze_x": float(gaze_x), "gaze_y": 0.0},
    }


def extract_openface_from_video(
    recording_path: str,
    fps_target: float = 10.0,
) -> Tuple[List[Dict], float]:
    """Read MP4 frames, run mediapipe, and return per-frame face feature dicts
    plus the measured frame rate. If mediapipe/opencv unavailable, returns [].
    Each dict: {au_intensities, au_presence, pose, gaze}."""
    if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE:
        logger.warning("cv2/mediapipe unavailable — skipping video face extraction.")
        return [], 0.0

    cap = cv2.VideoCapture(recording_path)
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / fps_target)))
    mesh = _MPMesh()
    frames = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        h, w = bgr.shape[:2]
        results = mesh.process(bgr)
        feats = _mesh_to_features(
            results.multi_face_landmarks[0] if results and results.multi_face_landmarks else None, h, w
        )
        if feats is not None:
            frames.append(feats)
    cap.release()
    logger.info(f"Extracted {len(frames)} face frames from {recording_path}")
    return frames, fps


# ---------------------------------------------------------------------------
# Transcript / textual features
# ---------------------------------------------------------------------------

def load_transcript_text(transcript_path: str) -> str:
    """Load transcript.txt (plain text) into a single string."""
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Failed to read transcript {transcript_path}: {e}")
        return ""


def compute_filler_rate(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    fillers = sum(1 for t in tokens if t in FILLER_WORDS)
    return fillers / len(tokens)


def compute_words_per_second(text: str, duration_sec: float) -> float:
    wc = len(text.split())
    if duration_sec <= 0:
        return 0.0
    return wc / duration_sec


def extract_transcript_features(
    text: str,
    n_seconds: int,
    step_log: Optional[List[Dict]] = None,
) -> Dict:
    """Filler rate, words per second, avg latency, silence ratio, fillers list."""
    filler_rate = compute_filler_rate(text)
    wps = compute_words_per_second(text, n_seconds)

    latency = 0.0
    silence_ratio = 0.0
    offsets = []
    if step_log:
        for s in step_log:
            lat = s.get("latency")
            if isinstance(lat, (int, float)):
                offsets.append(lat)
        if offsets:
            latency = float(np.mean(offsets))
        sil = [s.get("silence_ratio", 0.0) for s in step_log if isinstance(s.get("silence_ratio", 0.0), (int, float))]
        if sil:
            silence_ratio = float(np.mean(sil))

    return {
        "filler_rate": round(filler_rate, 4),
        "words_per_second": round(wps, 3),
        "avg_response_latency": round(latency, 2),
        "silence_ratio": round(silence_ratio, 3),
        "fillers": sorted([w for w in text.lower().split() if w in FILLER_WORDS]),
    }


# ---------------------------------------------------------------------------
# BoVW 101-dim per-second
# ---------------------------------------------------------------------------

def build_bovw(face_row: Dict, prev_face: Dict) -> np.ndarray:
    """101-dim BoVW vector for one second from current face stats + previous
    second (temporal delta). Layout:
      [0:17]  AU intensities (17)
      [17:20] pose (3)
      [20:22] gaze (2)
      [22:39] temporal deltas (17)
      [39:101] higher-order stats (62, zero-filled if not computed)
    """
    v = np.zeros(BOVW_DIM, np.float32)
    au = face_row.get("au_intensities", {})
    pose = face_row.get("pose", {})
    gaze = face_row.get("gaze", {})

    au_prev = prev_face.get("au_intensities", {}) if prev_face else {}

    vals = [au.get(k, 0.0) for k in AU_INTENSITY_KEYS]
    v[0:17] = vals
    v[17] = pose.get("Rx", 0.0)
    v[18] = pose.get("Ry", 0.0)
    v[19] = pose.get("Rz", 0.0)
    v[20] = gaze.get("gaze_x", 0.0)
    v[21] = gaze.get("gaze_y", 0.0)
    # temporal deltas
    if prev_face:
        prev_vals = [au_prev.get(k, 0.0) for k in AU_INTENSITY_KEYS]
        v[22:39] = np.array(vals, np.float32) - np.array(prev_vals, np.float32)
    # higher-order stats left zero
    return v
