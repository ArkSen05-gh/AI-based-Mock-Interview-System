"""
Feature Extractor — COVAREP / OpenFace / OpenSMILE eGeMAPS / Formant
Approximations using librosa + mediapipe landmarks.

For live WS sessions: derives proxy features from VoiceAnalyzer + FaceAnalyzer outputs.
For offline batch: reads raw WAV + CLNF_AUs.txt files (full COVAREP 74-dim, 17 AU).
"""
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("feature_extractor")

FILLER_WORDS = {
    "um", "uh", "like", "you", "know", "so",
    "actually", "basically", "erm", "ah", "hmm",
}

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


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


# ---- Live proxy features from analyzer outputs ---- #

def live_covarep_vector(voice_metrics: dict, stress_norm: float = 0.0) -> np.ndarray:
    """Build a 74-dim COVAREP-style vector from VoiceAnalyzer metrics.
    Layout mirrors the COVAREP CSV columns used in the paper:
    [0] f0_mean, [1] f0_std, [2] vuv_ratio, [3] naq, [4] qoq,
    [5] h1h2, [6] shimmer, [7] jitter,
    [8..32] 25 MCEP mean, [33..57] 25 HMPDM,
    [58..70] 13 HMPDD, [71] energy_rms, [72] speech_rate, [73] stress_norm
    """
    p = voice_metrics.get("pitch", 140.0)
    pv = voice_metrics.get("pitch_variance", 0.15)
    sr = voice_metrics.get("speech_rate", 3.0)
    trem = float(voice_metrics.get("voice_tremor", False))
    er = voice_metrics.get("energy_rms", 0.5)

    cov = np.zeros(74, np.float32)
    cov[0] = p / 400.0
    cov[1] = pv
    cov[2] = min(sr / 8.0, 1.0)
    cov[3] = er * pv * 0.5
    cov[4] = pv * 0.3
    cov[5] = min(1.0, pv * 2.0)
    cov[6] = pv * 0.8
    cov[7] = pv * 0.6 + float(trem) * 0.4

    for i in range(8, 33):
        cov[i] = np.sin(p * (i - 7) * np.pi / 400) * er
    for i in range(33, 58):
        cov[i] = np.cos(p * (i - 32) * np.pi / 400) * er * 0.8
    for i in range(58, 71):
        cov[i] = np.sin(p * (i - 57) * np.pi / 200) * er * 0.6

    cov[71] = er
    cov[72] = min(sr / 8.0, 1.0)
    cov[73] = stress_norm
    return cov


def live_openface_vector(face_metrics: dict) -> dict:
    """Derive OpenFace-style AU intensities + pose from FaceAnalyzer outputs."""
    t = face_metrics.get("facial_tension", 0.2)
    e = face_metrics.get("eye_contact_score", 75) / 100.0
    bs = face_metrics.get("blink_score", 80) / 100.0
    hm = face_metrics.get("head_movement_score", 90) / 100.0

    au_intensities = {
        "AU01_r": round(t * 3.0, 3),
        "AU02_r": round(max(0, (1 - t) * 2.0), 3),
        "AU04_r": round(t * 5.0, 3),
        "AU05_r": round(max(0, (1 - bs) * 2.0), 3),
        "AU06_r": round(max(0, (1 - t) * 4.0), 3),
        "AU07_r": round(t * 2.5, 3),
        "AU09_r": round(t * 3.5, 3),
        "AU10_r": round(t * 2.0, 3),
        "AU12_r": round(max(0, (1 - t) * 5.0), 3),
        "AU14_r": round(t * 1.5, 3),
        "AU15_r": round(t * 1.0, 3),
        "AU17_r": round(t * 2.0, 3),
        "AU20_r": round(t * 1.5, 3),
        "AU23_r": round(t * 2.5, 3),
        "AU24_r": round(t * 2.0, 3),
        "AU25_r": round(max(0, (1 - t) * 3.0), 3),
        "AU26_r": round(t * 1.0, 3),
    }
    au_presence = {
        "AU01_c": int(t > 0.3),
        "AU04_c": int(t > 0.5),
        "AU06_c": int(t < 0.5),
        "AU12_c": int(t < 0.4),
        "AU45_c": int(bs < 0.6),
        "AU20_c": int(t > 0.6),
    }
    rx = (1 - hm) * 15.0
    pose = {
        "Rx": round(rx, 2),
        "Ry": 0.0,
        "Rz": 0.0,
    }
    gaze = {"gaze_x": round(e - 0.5, 3), "gaze_y": 0.0}

    return {
        "au_intensities": au_intensities,
        "au_presence": au_presence,
        "pose": pose,
        "gaze": gaze,
    }


def live_egemaps_vector(voice_metrics: dict) -> np.ndarray:
    """Approximate OpenSMILE eGeMAPS 88-dim from voice metrics.
    Maps core eGeMAPS feature groups to approximate values."""
    p = voice_metrics.get("pitch", 140.0)
    pv = voice_metrics.get("pitch_variance", 0.15)
    sr = voice_metrics.get("speech_rate", 3.0)
    er = voice_metrics.get("energy_rms", 0.5)
    trem = float(voice_metrics.get("voice_tremor", False))

    v = np.zeros(88, np.float32)
    v[0] = p / 400.0
    v[1] = pv
    v[2] = sr / 8.0
    v[3] = er
    v[4] = float(trem)
    for i in range(5, 88):
        v[i] = np.sin(p * i * np.pi / 400) * er * 0.3
    return v


def live_formant_vector(voice_metrics: dict) -> np.ndarray:
    """F1-F5 approximations from pitch."""
    p = voice_metrics.get("pitch", 140.0)
    return np.array([p * k / 1000.0 for k in range(1, 6)], np.float32)


# ---- Offline full extraction from raw audio ---- #

def extract_covarep_from_wav(
    wav_path: str, sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """Extract per-second COVAREP-style 74-dim vectors from a WAV file.
    Returns (n_seconds, 74) and sample rate."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for offline COVAREP extraction")

    y, orig_sr = librosa.load(wav_path, sr=sr, mono=True)
    duration = len(y) / sr
    n_sec = int(np.ceil(duration))

    frames_per_sec = sr
    cov_rows = []

    for sec in range(n_sec):
        start = sec * sr
        end = min(start + sr, len(y))
        seg = y[start:end]
        if len(seg) < sr // 4:
            cov_rows.append(np.zeros(74, np.float32))
            continue

        try:
            f0 = librosa.pyin(seg, fmin=75, fmax=400, sr=sr, frame_length=2048)
            f0_vals = f0[0]
            f0_valid = f0_vals[~np.isnan(f0_vals) & (f0_vals > 0)]
        except Exception:
            f0_valid = np.array([140.0])

        if len(f0_valid) == 0:
            f0_valid = np.array([140.0])

        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        vuv_ratio = len(f0_valid) / max(1, len(f0_valid) + int(len(f0_vals) - len(f0_valid)) if 'f0_vals' in dir() else 1)
        vuv_ratio = min(1.0, vuv_ratio)

        rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
        er = float(np.mean(rms)) if len(rms) > 0 else 0.5

        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=25)
        mcep_mean = np.mean(mfcc, axis=1)

        spec = np.abs(np.fft.rfft(seg))
        h1h2 = 0.0
        if len(spec) > 100:
            h1h2 = float(np.log(spec[10] / spec[20])) if spec[20] > 0 else 0.0

        diff_f0 = np.abs(np.diff(f0_valid))
        shimmer = float(np.mean(diff_f0) / np.mean(f0_valid)) if len(diff_f0) > 0 and np.mean(f0_valid) > 0 else 0.0
        jitter = float(np.std(diff_f0) / np.mean(f0_valid)) if len(diff_f0) > 0 and np.mean(f0_valid) > 0 else 0.0

        pv = f0_std / f0_mean if f0_mean > 0 else 0.15
        naq = er * pv * 0.5
        qoq = pv * 0.3

        cov = np.zeros(74, np.float32)
        cov[0] = f0_mean / 400.0
        cov[1] = min(1.0, pv * 5.0)
        cov[2] = vuv_ratio
        cov[3] = naq
        cov[4] = qoq
        cov[5] = np.clip(h1h2, -3, 3) / 3.0
        cov[6] = min(1.0, shimmer * 5.0)
        cov[7] = min(1.0, jitter * 5.0)
        cov[8:33] = np.clip(mcep_mean / 50.0, -3, 3).astype(np.float32)

        spec_d = librosa.feature.delta(mfcc)
        cov[33:58] = np.clip(np.mean(spec_d, axis=1) / 10.0, -3, 3).astype(np.float32)
        spec_dd = librosa.feature.delta(mfcc, order=2)
        cov[58:71] = np.clip(np.mean(spec_dd[:, :13], axis=1) / 10.0, -3, 3).astype(np.float32)
        cov[71] = er
        cov[72] = len(librosa.onset.onset_strength(y=seg, sr=sr)) / max(1, len(seg) / sr)
        cov[73] = 0.0

        cov_rows.append(cov)

    return np.stack(cov_rows), sr


def read_openface_clnf(path: str) -> Optional[Dict]:
    """Read a DAIC-WOZ style OpenFace CLNF_AUs.txt file.
    Returns dict with 'au_intensities' (per-frame), 'pose', 'gaze' arrays."""
    try:
        import csv
        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=" ")
            rows = list(reader)
        if not rows:
            return None
        au_keys = [k for k in rows[0].keys() if "_r" in k]
        pose_keys = ["pitch", "yaw", "roll"]
        gaze_keys = [k for k in rows[0].keys() if "gaze" in k.lower()]

        au_data = {}
        for k in au_keys:
            au_data[k] = [float(rows[i].get(k, 0)) for i in range(len(rows))]

        pose_data = {}
        for k in pose_keys:
            if k in rows[0]:
                pose_data[k] = [float(rows[i].get(k, 0)) for i in range(len(rows))]

        gaze_data = {}
        for k in gaze_keys:
            gaze_data[k] = [float(rows[i].get(k, 0)) for i in range(len(rows))]

        return {
            "au_intensities": au_data,
            "pose": pose_data,
            "gaze": gaze_data,
            "n_frames": len(rows),
        }
    except Exception as e:
        logger.error(f"Failed to read OpenFace CLNF file: {e}")
        return None


# ---- CSV-based feature extraction for participant data ---- #

def load_covarep_csv(path: str) -> Optional[Dict]:
    """Load a COVAREP CSV file (76 cols: frame_time + 75 features).
    Returns dict with numpy arrays for each feature group."""
    try:
        import csv as csv_mod
        with open(path, "r") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        if not rows:
            return None

        n = len(rows)
        features = {
            "frame_time": np.zeros(n, np.float64),
            "F0": np.zeros(n, np.float64),
            "VUV": np.zeros(n, np.float64),
            "NAQ": np.zeros(n, np.float64),
            "QOQ": np.zeros(n, np.float64),
            "H1H2": np.zeros(n, np.float64),
            "shimmer": np.zeros(n, np.float64),
            "jitter": np.zeros(n, np.float64),
            "MCEP": np.zeros((n, 25), np.float32),
            "HMPDM": np.zeros((n, 25), np.float32),
            "HMPDD": np.zeros((n, 13), np.float32),
        }
        for i, row in enumerate(rows):
            features["frame_time"][i] = float(row.get("frame_time", 0))
            features["F0"][i] = float(row.get("F0", 0))
            features["VUV"][i] = float(row.get("VUV", 0))
            features["NAQ"][i] = float(row.get("NAQ", 0))
            features["QOQ"][i] = float(row.get("QOQ", 0))
            features["H1H2"][i] = float(row.get("H1H2", 0))
            features["shimmer"][i] = float(row.get("shimmer", 0))
            features["jitter"][i] = float(row.get("jitter", 0))
            for j in range(25):
                features["MCEP"][i, j] = float(row.get(f"MCEP_{j}", 0))
            for j in range(25):
                features["HMPDM"][i, j] = float(row.get(f"HMPDM_{j}", 0))
            for j in range(13):
                features["HMPDD"][i, j] = float(row.get(f"HMPDD_{j}", 0))
        features["n_frames"] = n
        return features
    except Exception as e:
        logger.error(f"Failed to load COVAREP CSV: {e}")
        return None


def load_openface_csv(path: str) -> Optional[Dict]:
    """Load an OpenFace_AUs.csv or CLNF_AUs.txt file.
    Auto-detects comma vs tab delimiter. Returns per-frame AU, pose, gaze data."""
    try:
        with open(path, "r") as f:
            first_line = f.readline()
            delimiter = "\t" if "\t" in first_line else ","

        import csv as csv_mod
        with open(path, "r") as f:
            reader = csv_mod.DictReader(f, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            return None

        n = len(rows)
        result = {"n_frames": n, "timestamps": np.zeros(n, np.float64)}

        au_keys_r = sorted([k for k in rows[0].keys() if k.endswith("_r")])
        au_keys_c = sorted([k for k in rows[0].keys() if k.endswith("_c")])

        for k in au_keys_r:
            result[k] = np.zeros(n, np.float32)
        for k in au_keys_c:
            result[k] = np.zeros(n, np.float32)

        pose_keys = ["pose_Rx", "pose_Ry", "pose_Rz"]
        gaze_keys = ["gaze_angle_x", "gaze_angle_y"]
        for k in pose_keys + gaze_keys:
            if k in rows[0]:
                result[k] = np.zeros(n, np.float32)

        for i, row in enumerate(rows):
            result["timestamps"][i] = float(row.get("timestamp", 0))
            for k in au_keys_r:
                try:
                    result[k][i] = float(row.get(k, 0))
                except (ValueError, KeyError):
                    pass
            for k in au_keys_c:
                try:
                    result[k][i] = float(row.get(k, 0))
                except (ValueError, KeyError):
                    pass
            for k in pose_keys + gaze_keys:
                if k in row:
                    try:
                        result[k][i] = float(row.get(k, 0))
                    except (ValueError, KeyError):
                        pass

        result["au_keys_r"] = au_keys_r
        result["au_keys_c"] = au_keys_c
        return result
    except Exception as e:
        logger.error(f"Failed to load OpenFace CSV: {e}")
        return None


def load_transcript_csv(path: str) -> Optional[List[Dict]]:
    """Load a Transcript CSV with columns: Start_Time, End_Time, Text, Confidence."""
    try:
        import csv as csv_mod
        with open(path, "r") as f:
            reader = csv_mod.DictReader(f)
            entries = []
            for row in reader:
                text = row.get("Text", row.get("text", "")).strip()
                if not text:
                    continue
                entries.append({
                    "start": float(row.get("Start_Time", row.get("start_time", 0))),
                    "end": float(row.get("End_Time", row.get("end_time", 0))),
                    "text": text,
                    "confidence": float(row.get("Confidence", row.get("confidence", 0.9))),
                })
        return entries if entries else None
    except Exception as e:
        logger.error(f"Failed to load transcript CSV: {e}")
        return None


def load_bovw_csv(path: str) -> Optional[Dict]:
    """Load a BoVW_Pose_Gaze_AUs.csv file (101-dim aggregated features per second).
    Returns dict with timestamps and feature matrix."""
    try:
        import csv as csv_mod
        with open(path, "r") as f:
            reader = csv_mod.reader(f)
            rows = list(reader)
        if not rows:
            return None

        n = len(rows)
        data = np.zeros((n, 101), np.float32)
        timestamps = np.zeros(n, np.float64)
        for i, row in enumerate(rows):
            vals = [float(x) for x in row]
            if len(vals) >= 101:
                timestamps[i] = vals[1] if len(vals) > 1 else 0
                data[i] = np.array(vals[1:102], np.float32) if len(vals) >= 102 else np.array(vals[1:], np.float32)[:101]
            elif len(vals) > 1:
                timestamps[i] = vals[1] if len(vals) > 1 else 0
                data[i, :len(vals)-1] = np.array(vals[1:], np.float32)
        return {"timestamps": timestamps, "features": data, "n_rows": n}
    except Exception as e:
        logger.error(f"Failed to load BoVW CSV: {e}")
        return None


def aggregate_to_seconds(timestamps: np.ndarray, values: np.ndarray, n_seconds: int = None) -> np.ndarray:
    """Aggregate frame-level data to per-second averages.
    timestamps: (N,) array of timestamps in seconds
    values: (N,) or (N, D) array of values
    Returns: (n_seconds, D) array of per-second means."""
    if n_seconds is None:
        n_seconds = int(np.ceil(timestamps.max())) + 1

    if values.ndim == 1:
        result = np.zeros(n_seconds, np.float64)
    else:
        result = np.zeros((n_seconds, values.shape[1]), np.float64)

    for sec in range(n_seconds):
        mask = (timestamps >= sec) & (timestamps < sec + 1)
        if not np.any(mask):
            if sec > 0:
                result[sec] = result[sec - 1]
            continue
        if values.ndim == 1:
            result[sec] = np.mean(values[mask])
        else:
            result[sec] = np.mean(values[mask], axis=0)
    return result


def extract_voice_features_from_covarep(cov_features: Dict) -> List[Dict]:
    """Convert COVAREP features to per-second voice metric dicts compatible with the pipeline.
    Returns list of dicts, one per second."""
    ft = cov_features["frame_time"]
    raw_f0 = cov_features["F0"]
    n_seconds = int(np.ceil(ft.max())) + 1

    f0_sec = aggregate_to_seconds(ft, raw_f0, n_seconds)
    vuv_sec = aggregate_to_seconds(ft, cov_features["VUV"], n_seconds)
    naq_sec = aggregate_to_seconds(ft, cov_features["NAQ"], n_seconds)
    h1h2_sec = aggregate_to_seconds(ft, cov_features["H1H2"], n_seconds)
    shimmer_sec = aggregate_to_seconds(ft, cov_features["shimmer"], n_seconds)
    jitter_sec = aggregate_to_seconds(ft, cov_features["jitter"], n_seconds)

    mcep_sec = aggregate_to_seconds(ft, cov_features["MCEP"], n_seconds)

    voice_per_sec = []
    for s in range(n_seconds):
        pitch = float(f0_sec[s]) if f0_sec[s] > 0 else 140.0

        # FIX 1 — F0 std computed over raw per-frame F0 values within THIS
        # 1-second window, NOT a rolling std over aggregated per-second means.
        f0_in_window = raw_f0[(ft >= s) & (ft < s + 1)]
        f0_valid = f0_in_window[f0_in_window > 0]
        f0_std_sec = float(np.std(f0_valid)) if len(f0_valid) > 1 else 15.0
        pv = f0_std_sec / max(pitch, 1.0)

        # FIX 2 — VUV = 1 means VOICED (speech present). do NOT invert.
        #   CRITICAL: speech_rate = VUV_ratio * 5.0
        sr_est = float(vuv_sec[s]) * 5.0
        er = float(np.mean(np.abs(mcep_sec[s]))) / 20.0 if np.any(mcep_sec[s]) else 0.5
        tremor = bool(shimmer_sec[s] > 0.15 or jitter_sec[s] > 0.1)

        voice_per_sec.append({
            "pitch": round(pitch, 2),
            "pitch_variance": round(min(1.0, max(0.0, pv)), 4),
            "speech_rate": round(max(0.0, sr_est), 2),
            "voice_tremor": tremor,
            "energy_rms": round(max(0.01, min(1.0, er)), 3),
            "audio_stress_score": round(min(100.0, max(0.0, pv * 60 + float(naq_sec[s]) * 20)), 2),
            "pause_duration": round(float(vuv_sec[s]) * 5.0, 2),
            "filler_rate": 0.0,
            "f0_mean": round(pitch, 2),
            "f0_std": round(f0_std_sec, 4),
            "vuv_ratio": round(float(vuv_sec[s]), 4),
            "naq": round(float(naq_sec[s]), 4),
            "h1h2": round(float(h1h2_sec[s]), 4),
            "shimmer": round(float(shimmer_sec[s]), 4),
            "jitter": round(float(jitter_sec[s]), 4),
        })
    return voice_per_sec


def extract_face_features_from_openface(oface_data: Dict) -> List[Dict]:
    """Convert OpenFace CLNF data to per-second face metric dicts.
    Returns list of dicts, one per second."""
    timestamps = oface_data["timestamps"]
    n_seconds = int(np.ceil(timestamps.max())) + 1

    au_r_keys = oface_data.get("au_keys_r", [])
    au_c_keys = oface_data.get("au_keys_c", [])

    face_per_sec = []
    for s in range(n_seconds):
        mask = (timestamps >= s) & (timestamps < s + 1)
        if not np.any(mask):
            if face_per_sec:
                face_per_sec.append(face_per_sec[-1])
            else:
                face_per_sec.append({
                    "eye_contact_score": 75.0, "blink_score": 80.0,
                    "head_movement_score": 90.0, "facial_tension": 0.2,
                })
            continue

        au12_r = float(np.mean(oface_data.get("AU12_r", np.zeros(1))[mask])) if "AU12_r" in oface_data else 0.5
        au04_r = float(np.mean(oface_data.get("AU04_r", np.zeros(1))[mask])) if "AU04_r" in oface_data else 0.2
        au06_r = float(np.mean(oface_data.get("AU06_r", np.zeros(1))[mask])) if "AU06_r" in oface_data else 0.5
        au01_r = float(np.mean(oface_data.get("AU01_r", np.zeros(1))[mask])) if "AU01_r" in oface_data else 0.2
        au07_r = float(np.mean(oface_data.get("AU07_r", np.zeros(1))[mask])) if "AU07_r" in oface_data else 0.2
        au15_r = float(np.mean(oface_data.get("AU15_r", np.zeros(1))[mask])) if "AU15_r" in oface_data else 0.1
        au20_r = float(np.mean(oface_data.get("AU20_r", np.zeros(1))[mask])) if "AU20_r" in oface_data else 0.1
        au25_r = float(np.mean(oface_data.get("AU25_r", np.zeros(1))[mask])) if "AU25_r" in oface_data else 0.1

        au45_c = 0
        if "AU45_c" in oface_data:
            au45_c = int(np.mean(oface_data["AU45_c"][mask]) > 0.5)

        rx = 0.0
        if "pose_Rx" in oface_data:
            rx = float(np.mean(oface_data["pose_Rx"][mask]))
        ry = 0.0
        if "pose_Ry" in oface_data:
            ry = float(np.mean(oface_data["pose_Ry"][mask]))
        rz = 0.0
        if "pose_Rz" in oface_data:
            rz = float(np.mean(oface_data["pose_Rz"][mask]))

        gaze_x = 0.0
        if "gaze_angle_x" in oface_data:
            gaze_x = float(np.mean(oface_data["gaze_angle_x"][mask]))

        eye_contact = max(0, min(100, 50 + (1 - abs(gaze_x)) * 50))
        blink_score = max(0, min(100, 70 + au45_c * 30))
        head_movement = max(0, min(100, 90 - abs(rx) * 20 - abs(ry) * 10))
        tension = max(0, min(1, (au04_r + au07_r + au15_r) / 3.0 * 2.0))

        face_per_sec.append({
            "eye_contact_score": round(eye_contact, 2),
            "blink_score": round(blink_score, 2),
            "head_movement_score": round(head_movement, 2),
            "facial_tension": round(tension, 3),
            "au12_r": round(au12_r, 3),
            "au04_r": round(au04_r, 3),
            "au06_r": round(au06_r, 3),
            "au01_r": round(au01_r, 3),
            "au07_r": round(au07_r, 3),
            "au15_r": round(au15_r, 3),
            "au20_r": round(au20_r, 3),
            "au25_r": round(au25_r, 3),
            "au45_c": au45_c,
            "pose_rx": round(rx, 4),
            "pose_ry": round(ry, 4),
            "pose_rz": round(rz, 4),
            "gaze_x": round(gaze_x, 4),
        })
    return face_per_sec


def build_participant_features(participant_dir: str) -> Optional[Dict]:
    """Load all CSVs from a participant directory and build per-second feature arrays.
    Returns dict with voice_per_sec, face_per_sec, transcript_text, n_seconds."""
    import glob as glob_mod
    pid = os.path.basename(participant_dir.rstrip("/\\"))

    cov_files = glob_mod.glob(os.path.join(participant_dir, f"*COVAREP*"))
    oface_files = glob_mod.glob(os.path.join(participant_dir, f"*OpenFace_AUs*"))
    clnf_files = glob_mod.glob(os.path.join(participant_dir, f"*CLNF_AU*"))
    transcript_files = glob_mod.glob(os.path.join(participant_dir, f"*Transcript*"))
    bovw_files = glob_mod.glob(os.path.join(participant_dir, f"*BoVW*"))

    voice_per_sec = []
    face_per_sec = []
    transcript_text = ""
    n_seconds = 0

    if cov_files:
        cov = load_covarep_csv(cov_files[0])
        if cov:
            voice_per_sec = extract_voice_features_from_covarep(cov)
            n_seconds = max(n_seconds, len(voice_per_sec))
            logger.info(f"[{pid}] COVAREP: {cov['n_frames']} frames -> {len(voice_per_sec)} seconds")

    face_file = oface_files[0] if oface_files else (clnf_files[0] if clnf_files else None)
    if face_file:
        oface = load_openface_csv(face_file)
        if oface:
            face_per_sec = extract_face_features_from_openface(oface)
            n_seconds = max(n_seconds, len(face_per_sec))
            logger.info(f"[{pid}] OpenFace: {oface['n_frames']} frames -> {len(face_per_sec)} seconds")

    if transcript_files:
        entries = load_transcript_csv(transcript_files[0])
        if entries:
            transcript_text = " ".join(e["text"] for e in entries)
            if not voice_per_sec:
                max_t = max(e["end"] for e in entries)
                n_seconds = max(n_seconds, int(np.ceil(max_t)))
            logger.info(f"[{pid}] Transcript: {len(entries)} segments")

    if not voice_per_sec:
        voice_per_sec = [{"pitch": 140, "pitch_variance": 0.15, "speech_rate": 3.0,
                          "voice_tremor": False, "energy_rms": 0.5, "audio_stress_score": 20,
                          "pause_duration": 0.5, "filler_rate": 0.0}
                         for _ in range(max(n_seconds, 30))]
        n_seconds = max(n_seconds, 30)

    if not face_per_sec:
        face_per_sec = [{"eye_contact_score": 75, "blink_score": 80,
                         "head_movement_score": 90, "facial_tension": 0.2}
                        for _ in range(n_seconds)]

    filler_rate = compute_filler_rate(transcript_text)
    for v in voice_per_sec:
        v["filler_rate"] = filler_rate

    while len(voice_per_sec) < n_seconds:
        voice_per_sec.append(voice_per_sec[-1] if voice_per_sec else {
            "pitch": 140, "pitch_variance": 0.15, "speech_rate": 3.0,
            "voice_tremor": False, "energy_rms": 0.5, "audio_stress_score": 20,
            "pause_duration": 0.5, "filler_rate": filler_rate
        })
    while len(face_per_sec) < n_seconds:
        face_per_sec.append(face_per_sec[-1] if face_per_sec else {
            "eye_contact_score": 75, "blink_score": 80,
            "head_movement_score": 90, "facial_tension": 0.2
        })

    return {
        "pid": pid,
        "voice_per_sec": voice_per_sec[:n_seconds],
        "face_per_sec": face_per_sec[:n_seconds],
        "transcript_text": transcript_text,
        "n_seconds": n_seconds,
    }
