"""
Session Analyzer — runs the four core research models on a session's
extracted per-second features and produces the full results dict used by the
results page, the four plots, and the recommendation report.

Mirrors `run_offline_from_participant_dir` in offline_analysis.py but consumes
features extracted from an actual session recording (offline_extractor) rather
than a participant data directory.
"""
import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger("session_analyzer")

from research.offline_extractor import (
    extract_audio,
    extract_covarep_from_audio,
    extract_egemaps_from_audio,
    extract_formants_from_audio,
    extract_openface_from_video,
    build_bovw,
    load_transcript_text,
    extract_transcript_features,
    AU_INTENSITY_KEYS,
    COVAREP_DIM,
    BOVW_DIM,
)
from research.interview_engine import InterviewEngine


def _covarep_row_to_voice(row: np.ndarray) -> Dict:
    """Convert one 74-dim COVAREP row to the live-style voice metrics dict."""
    f0 = float(row[0]) * 400.0
    if not f0 or f0 < 1:
        f0 = 140.0
    pv = float(row[1]) / 5.0
    vuv = float(np.clip(row[2], 0, 1))
    naq = float(row[3])
    shim = float(row[6])
    jit = float(row[7])
    er = float(np.clip(row[71] * 2.0, 0.01, 1.0))

    sr = vuv * 5.0
    return {
        "pitch": round(f0, 2),
        "pitch_variance": round(min(1.0, max(0.0, pv)), 4),
        "speech_rate": round(max(0.0, sr), 2),
        "voice_tremor": bool(shim > 0.15 or jit > 0.1),
        "energy_rms": round(max(0.01, min(1.0, er)), 3),
        "audio_stress_score": round(min(100.0, max(0.0, pv * 60 + naq * 20)), 2),
        "pause_duration": round(float(vuv) * 5.0, 2),
        "filler_rate": 0.0,
    }


def _row_to_face(row: Dict, prev: Dict) -> Dict:
    """Convert one per-second mediapipe face summary dict to the live-style
    face metrics dict used by the models."""
    au = row.get("au_intensities", {})
    pose = row.get("pose", {})
    gaze = row.get("gaze", {})
    au_pres = row.get("au_presence", {})

    au04 = au.get("AU04_r", 0.0)
    au07 = au.get("AU07_r", 0.0)
    au15 = au.get("AU15_r", 0.0)
    au12 = au.get("AU12_r", 0.0)
    au45_c = int(au_pres.get("AU45_c", 0))

    rx = pose.get("Rx", 0.0)
    ry = pose.get("Ry", 0.0)
    gz = gaze.get("gaze_x", 0.0)

    tension = max(0.0, min(1.0, (au04 + au07 + au15) / 3.0 * 2.0))
    eye_contact = max(0, min(100, 50 + (1 - abs(gz)) * 50))
    blink = max(0, min(100, 70 + au45_c * 30))
    head_move = max(0, min(100, 90 - abs(rx) * 20 - abs(ry) * 10))

    return {
        "eye_contact_score": round(eye_contact, 2),
        "blink_score": round(blink, 2),
        "head_movement_score": round(head_move, 2),
        "facial_tension": round(tension, 3),
        "au12_r": round(au12, 3),
        "au04_r": round(au04, 3),
        "au45_c": au45_c,
        "pose_rx": round(rx, 4),
    }


def _average_face_frames(frames: List[Dict], sec: float, fps_target: float) -> Optional[Dict]:
    """Average the raw mediapipe frame dicts within [sec, sec+1). Returns a
    per-second summary dict, or None if no face frames in this second."""
    lo, hi = sec, sec + 1.0
    win = [f for f in frames if lo <= f["t"] < hi]
    if not win:
        return None

    def mean_au(key):
        vals = [f["au_intensities"].get(key, 0.0) for f in win]
        return float(np.mean(vals))

    au_sum = {k: mean_au(k) for k in AU_INTENSITY_KEYS}

    def mean_pres(key):
        vals = [f["au_presence"].get(key, 0) for f in win]
        return int(np.mean(vals) > 0.5)

    au_pres = {k: mean_pres(k) for k in ["AU01_c", "AU04_c", "AU06_c", "AU12_c", "AU45_c"]}

    pose = {
        "Rx": float(np.mean([f["pose"]["Rx"] for f in win])),
        "Ry": float(np.mean([f["pose"]["Ry"] for f in win])),
        "Rz": float(np.mean([f["pose"]["Rz"] for f in win])),
    }
    gaze = {
        "gaze_x": float(np.mean([f["gaze"]["gaze_x"] for f in win])),
        "gaze_y": float(np.mean([f["gaze"]["gaze_y"] for f in win])),
    }
    return {"au_intensities": au_sum, "au_presence": au_pres, "pose": pose, "gaze": gaze}


def analyze_session(
    session_id: str,
    sessions_dir: str = "sessions",
    domain: str = "general",
    calibration: Optional[Dict] = None,
) -> Dict:
    """Run the 4-model pipeline on a session directory:
    sessions/<session_id>/ containing recording.mp4 (+ optional transcript.txt,
    answer_started_at.txt, step_log.json). Returns the full results dict."""
    session_dir = os.path.join(sessions_dir, session_id)
    if not os.path.isdir(session_dir):
        return {"error": f"Session directory not found: {session_dir}"}

    recording = None
    for name in ("recording.mp4", "recording.webm", "audio.wav", "recording.wav"):
        p = os.path.join(session_dir, name)
        if os.path.exists(p):
            recording = p
            break

    y = extract_audio(session_dir, recording)
    n_seconds = 0
    if y is not None:
        n_seconds = int(np.ceil(len(y) / 16000))

    covarep = extract_covarep_from_audio(y) if y is not None else None
    egemaps = extract_egemaps_from_audio(y) if y is not None else None
    formants = extract_formants_from_audio(y) if y is not None else None

    cv_available = covarep is not None and covarep.shape[0] > 0
    n_seconds = max(n_seconds, cv_available * covarep.shape[0])

    # Video face frames
    video_frames = []
    if recording and recording.lower().endswith((".mp4", ".webm")):
        frames, fps = extract_openface_from_video(recording)
        for i, f in enumerate(frames):
            f["t"] = i / max(1.0, fps_target_from(fps))
        video_frames = frames
    elif recording and recording.lower().endswith((".wav",)):
        pass

    # Build per-second face summaries
    face_summaries = []
    for sec in range(n_seconds):
        fs = _average_face_frames(video_frames, sec, 10.0)
        face_summaries.append(fs)

    # Transcript
    transcript_path = os.path.join(session_dir, "transcript.txt")
    text = load_transcript_text(transcript_path)
    step_log = _load_step_log(session_dir)
    trans_feats = extract_transcript_features(text, n_seconds, step_log)

    # Build BoVW 101-dim
    bovw_rows = []
    prev = None
    for fs in face_summaries:
        if fs is None:
            fs = {"au_intensities": {}, "au_presence": {}, "pose": {}, "gaze": {}}
        bovw_rows.append(build_bovw(fs, prev or fs))
        prev = fs

    # ---- Persist extracted CSV/JSON artifacts under the session dir ----
    persist_dir = session_dir
    _persist_artifacts(
        persist_dir, covarep, egemaps, formants, np.array(bovw_rows) if bovw_rows else None,
        trans_feats, face_summaries,
    )

    # ---- Run the online-equivalent engine over each second ----
    engine = InterviewEngine(domain=domain, session_id=session_id, calibration=calibration)
    results = []
    for sec in range(n_seconds):
        voice = _covarep_row_to_voice(covarep[sec]) if cv_available else {
            "pitch": 140.0, "pitch_variance": 0.15, "speech_rate": 3.0,
            "voice_tremor": False, "energy_rms": 0.5, "audio_stress_score": 20.0,
            "pause_duration": 0.5, "filler_rate": trans_feats["filler_rate"],
        }
        fs = face_summaries[sec]
        if fs is None:
            face = {"eye_contact_score": 75.0, "blink_score": 80.0,
                    "head_movement_score": 90.0, "facial_tension": 0.2}
        else:
            face = _row_to_face(fs, face_summaries[sec - 1] if sec > 0 else None)
        result = engine.step(face=face, voice=voice)
        if result:
            results.append(result)

    summary = engine.finalize()
    summary["per_second"] = results
    summary["session_id"] = session_id
    summary["transcript_features"] = trans_feats
    summary["artifacts_dir"] = persist_dir

    _save_summary(session_dir, summary)
    return summary


def fps_target_from(fps: float) -> float:
    return max(1.0, fps or 30.0)


def _load_step_log(session_dir: str) -> List[Dict]:
    p = os.path.join(session_dir, "step_log.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_summary(session_dir: str, summary: Dict):
    out = {k: v for k, v in summary.items() if k != "per_second"}
    out["n_per_second"] = len(summary.get("per_second", []))
    path = os.path.join(session_dir, "results.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not save results.json: {e}")


def _persist_artifacts(
    persist_dir: str,
    covarep: Optional[np.ndarray],
    egemaps: Optional[np.ndarray],
    formants: Optional[np.ndarray],
    bovw: Optional[np.ndarray],
    trans_feats: Dict,
    face_summaries: List[Optional[Dict]],
):
    os.makedirs(persist_dir, exist_ok=True)

    def write_csv(name, arr):
        if arr is None:
            return
        path = os.path.join(persist_dir, name)
        try:
            np.savetxt(path, arr, delimiter=",")
        except Exception as e:
            logger.warning(f"Failed to write {name}: {e}")

    def write_json(name, obj):
        path = os.path.join(persist_dir, name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write {name}: {e}")

    write_csv("covarep.csv", covarep)
    write_csv("egemaps_features.csv", egemaps)
    write_csv("formants.csv", formants)
    write_csv("bovw_features.csv", bovw)
    write_json("transcript_features.json", trans_feats)

    face_rows = []
    for i, fs in enumerate(face_summaries):
        row = {"second": i}
        if fs:
            for k, v in fs.get("au_intensities", {}).items():
                row[f"AU_{k}"] = round(v, 4)
            for k, v in fs.get("pose", {}).items():
                row[f"pose_{k}"] = round(v, 4)
            for k, v in fs.get("gaze", {}).items():
                row[f"gaze_{k}"] = round(v, 4)
            for k, v in fs.get("au_presence", {}).items():
                row[k] = v
        face_rows.append(row)
    write_json("openface.json", face_rows)
