"""
Offline Analysis — Batch-mode research pipeline with graph output.

Usage:
    python -m research.offline_analysis --data_dir ./backend/617_P/
    python -m research.offline_analysis --data_dir ./backend/  (all participants)
    python -m research.offline_analysis --demo   (simulated session)

Reads COVAREP CSV, OpenFace CLNF_AUs.txt/OpenFace_AUs.csv, Transcript CSV,
BoVW_Pose_Gaze_AUs.csv and runs the full BiLSTM + AU-CNN + FSM + PHQ-8 pipeline,
outputting report.json and stress/confidence/FSM graphs.
"""
import argparse
import json
import os
import sys
import time
import glob
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger("offline_analysis")

# Add parent dir so research package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.interview_engine import InterviewEngine
from research.feature_extractor import (
    live_covarep_vector,
    live_openface_vector,
    compute_filler_rate,
    build_participant_features,
)


def run_offline_from_files(
    covarep_path: Optional[str] = None,
    openface_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    bovw_path: Optional[str] = None,
    domain: str = "general",
) -> Dict:
    """Run the full research pipeline from recorded participant files.

    Now uses build_participant_features() to properly extract real features
    from COVAREP (F0, VUV, MCEP, etc.), OpenFace (AUs, pose, gaze),
    and transcripts — instead of synthetic mappings.
    """
    engine = InterviewEngine(domain=domain, session_id="offline_batch")

    participant_dir = None
    if covarep_path:
        participant_dir = os.path.dirname(covarep_path)
    elif openface_path:
        participant_dir = os.path.dirname(openface_path)
    elif transcript_path:
        participant_dir = os.path.dirname(transcript_path)

    if participant_dir:
        features = build_participant_features(participant_dir)
        if features:
            voice_per_sec = features["voice_per_sec"]
            face_per_sec = features["face_per_sec"]
            n_seconds = features["n_seconds"]

            logger.info(f"Loaded participant {features['pid']}: {n_seconds}s, "
                       f"{len(voice_per_sec)} voice frames, {len(face_per_sec)} face frames")

            results = []
            for sec in range(n_seconds):
                voice = voice_per_sec[sec] if sec < len(voice_per_sec) else voice_per_sec[-1]
                face = face_per_sec[sec] if sec < len(face_per_sec) else face_per_sec[-1]

                result = engine.step(face=face, voice=voice)
                if result:
                    results.append(result)

            summary = engine.finalize()
            summary["per_second"] = results
            summary["participant_id"] = features["pid"]
            return summary

    logger.warning("Could not load participant directory, falling back to file-level loading")

    cov_data = None
    if covarep_path and os.path.exists(covarep_path):
        from research.feature_extractor import load_covarep_csv
        cov = load_covarep_csv(covarep_path)
        if cov:
            from research.feature_extractor import aggregate_to_seconds
            ft = cov["frame_time"]
            n_seconds = int(np.ceil(ft.max())) + 1
            f0_sec = aggregate_to_seconds(ft, cov["F0"], n_seconds)
            vuv_sec = aggregate_to_seconds(ft, cov["VUV"], n_seconds)
            naq_sec = aggregate_to_seconds(ft, cov["NAQ"], n_seconds)
            mcep_sec = aggregate_to_seconds(ft, cov["MCEP"], n_seconds)
            shimmer_sec = aggregate_to_seconds(ft, cov["shimmer"], n_seconds)
            jitter_sec = aggregate_to_seconds(ft, cov["jitter"], n_seconds)
            h1h2_sec = aggregate_to_seconds(ft, cov["H1H2"], n_seconds)
            cov_data = {
                "f0": f0_sec, "vuv": vuv_sec, "naq": naq_sec,
                "mcep": mcep_sec, "shimmer": shimmer_sec, "jitter": jitter_sec,
                "h1h2": h1h2_sec, "n_seconds": n_seconds,
            }
            logger.info(f"Loaded COVAREP: {cov['n_frames']} frames -> {n_seconds} seconds")

    oface_data = None
    if openface_path and os.path.exists(openface_path):
        from research.feature_extractor import load_openface_csv
        oface_data = load_openface_csv(openface_path)
        if oface_data:
            logger.info(f"Loaded OpenFace: {oface_data['n_frames']} frames")

    transcript_text = ""
    if transcript_path and os.path.exists(transcript_path):
        from research.feature_extractor import load_transcript_csv
        entries = load_transcript_csv(transcript_path)
        if entries:
            transcript_text = " ".join(e["text"] for e in entries)

    filler_rate = compute_filler_rate(transcript_text)

    if cov_data is not None:
        n_seconds = cov_data["n_seconds"]
    elif oface_data is not None:
        n_seconds = int(np.ceil(oface_data["timestamps"].max())) + 1
    else:
        n_seconds = 30

    results = []
    for sec in range(n_seconds):
        if cov_data is not None and sec < n_seconds:
            f0 = float(cov_data["f0"][sec]) if cov_data["f0"][sec] > 0 else 140.0
            vuv = float(cov_data["vuv"][sec])
            naq = float(cov_data["naq"][sec])
            shim = float(cov_data["shimmer"][sec])
            jit = float(cov_data["jitter"][sec])
            h1h2 = float(cov_data["h1h2"][sec])

            f0_std = float(np.std(cov_data["f0"][max(0, sec-5):sec+1])) if sec > 0 else 15.0
            pv = f0_std / max(f0, 1.0)
            er = float(np.mean(np.abs(cov_data["mcep"][sec]))) / 20.0

            voice = {
                "pitch": round(f0, 2),
                "pitch_variance": round(min(1.0, pv), 4),
                "speech_rate": round(max(0.5, (1 - vuv) * 5.0), 2),
                "voice_tremor": bool(shim > 0.15 or jit > 0.1),
                "energy_rms": round(max(0.01, min(1.0, er)), 3),
                "audio_stress_score": round(min(100, max(0, pv * 60 + naq * 20)), 2),
                "pause_duration": round(vuv * 5.0, 2),
                "filler_rate": filler_rate,
            }
        else:
            voice = {
                "pitch": round(140 + np.random.normal(0, 10), 2),
                "pitch_variance": round(max(0, 0.15 + np.random.normal(0, 0.05)), 3),
                "speech_rate": round(3 + np.random.normal(0, 0.5), 2),
                "voice_tremor": bool(np.random.random() > 0.8),
                "energy_rms": round(max(0.1, 0.5 + np.random.normal(0, 0.1)), 3),
                "audio_stress_score": round(max(0, 20 + np.random.normal(0, 10)), 2),
                "pause_duration": round(max(0, np.random.exponential(0.5)), 2),
                "filler_rate": filler_rate,
            }

        if oface_data and "timestamps" in oface_data:
            ts = oface_data["timestamps"]
            mask = (ts >= sec) & (ts < sec + 1)
            if np.any(mask):
                def safe_mean(key):
                    if key in oface_data:
                        return float(np.mean(oface_data[key][mask]))
                    return 0.0

                au12 = safe_mean("AU12_r")
                au04 = safe_mean("AU04_r")
                au06 = safe_mean("AU06_r")
                au01 = safe_mean("AU01_r")
                au07 = safe_mean("AU07_r")
                au15 = safe_mean("AU15_r")
                au45_c = 0
                if "AU45_c" in oface_data:
                    au45_c = int(np.mean(oface_data["AU45_c"][mask]) > 0.5)
                rx = safe_mean("pose_Rx") if "pose_Rx" in oface_data else 0
                ry = safe_mean("pose_Ry") if "pose_Ry" in oface_data else 0
                gz = safe_mean("gaze_angle_x") if "gaze_angle_x" in oface_data else 0

                tension = max(0, min(1, (au04 + au07 + au15) / 3.0 * 2.0))
                eye_contact = max(0, min(100, 50 + (1 - abs(gz)) * 50))
                blink = max(0, min(100, 70 + au45_c * 30))
                head_move = max(0, min(100, 90 - abs(rx) * 20 - abs(ry) * 10))

                face = {
                    "eye_contact_score": round(eye_contact, 2),
                    "blink_score": round(blink, 2),
                    "head_movement_score": round(head_move, 2),
                    "facial_tension": round(tension, 3),
                    "au12_r": round(au12, 3),
                    "au04_r": round(au04, 3),
                    "au06_r": round(au06, 3),
                    "au45_c": au45_c,
                    "pose_rx": round(rx, 4),
                }
            else:
                face = {
                    "eye_contact_score": 75.0, "blink_score": 80.0,
                    "head_movement_score": 90.0, "facial_tension": 0.2,
                }
        else:
            face = {
                "eye_contact_score": round(75 + np.random.normal(0, 5), 2),
                "blink_score": round(80 + np.random.normal(0, 3), 2),
                "head_movement_score": round(90 + np.random.normal(0, 2), 2),
                "facial_tension": round(max(0, min(1, 0.25 + np.random.normal(0, 0.08))), 3),
            }

        result = engine.step(face=face, voice=voice)
        if result:
            results.append(result)

    summary = engine.finalize()
    summary["per_second"] = results
    return summary


def run_offline_from_participant_dir(
    participant_dir: str,
    domain: str = "general",
) -> Dict:
    """Run the pipeline on a full participant directory using build_participant_features."""
    engine = InterviewEngine(domain=domain, session_id="offline_batch")
    features = build_participant_features(participant_dir)

    if not features:
        logger.error(f"Failed to load features from {participant_dir}")
        return {"error": "Failed to load features"}

    voice_per_sec = features["voice_per_sec"]
    face_per_sec = features["face_per_sec"]
    n_seconds = features["n_seconds"]

    logger.info(f"Running offline analysis: {features['pid']} ({n_seconds}s)")

    results = []
    for sec in range(n_seconds):
        voice = voice_per_sec[sec]
        face = face_per_sec[sec]
        result = engine.step(face=face, voice=voice)
        if result:
            results.append(result)

    summary = engine.finalize()
    summary["per_second"] = results
    summary["participant_id"] = features["pid"]
    return summary


def run_all_participants(
    data_dir: str,
    domain: str = "general",
) -> Dict:
    """Run offline analysis on ALL participant directories and aggregate results.
    Uses the faster feature-based computation (calibrate_session_analytics) instead
    of the per-second InterviewEngine, to keep batch analysis tractable."""
    from training.pipeline import discover_participants, generate_pseudo_labels
    from research.feature_extractor import build_participant_features
    from research.session_analytics import resample_to_100

    from research.fsm_controller import FSMAdaptiveController
    from research.phq8_estimator import PHQ8Estimator

    participants = discover_participants(data_dir)

    if not participants:
        return {"error": "No participants found"}

    all_results = {}
    all_stress_curves = []
    all_conf_curves = []
    all_fsm_series = []

    STATE_LEVELS = ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"]

    for pdir in participants:
        pid = os.path.basename(pdir)
        logger.info(f"Processing {pid}...")
        try:
            features = build_participant_features(pdir)
            if not features:
                all_results[pid] = {"error": "No features"}
                continue

            voice_per_sec = features["voice_per_sec"]
            face_per_sec = features["face_per_sec"]
            n_seconds = features["n_seconds"]

            p_stress = []
            p_conf = []
            fsm_ctrl = FSMAdaptiveController(domain)
            fsm_series = []
            phq = PHQ8Estimator()

            for s in range(n_seconds):
                v = voice_per_sec[s]
                f = face_per_sec[s]
                stress_raw, conf = generate_pseudo_labels(v, f)
                p_stress.append(stress_raw)
                p_conf.append(conf)

                stress_norm = float(np.clip(stress_raw, 0, 1))
                conf_pct = float(np.clip(conf * 100, 0, 100))
                st = fsm_ctrl.step(stress_norm, conf_pct)
                fsm_series.append(st["fsm_state"])
                phq.add_frame(f, v, stress_norm, conf)

            stress_arr = np.array(p_stress, np.float32)
            conf_arr = np.array(p_conf, np.float32)

            import statistics
            filler = statistics.mean([v.get("filler_rate", 0) for v in voice_per_sec])

            phq_res = phq.predict()

            fsm_pct = {}
            for lvl in STATE_LEVELS:
                fsm_pct[lvl] = round(fsm_series.count(lvl) / max(1, n_seconds) * 100, 2)

            all_results[pid] = {
                "n_seconds": n_seconds,
                "stress_mean": round(float(stress_arr.mean()), 4),
                "stress_std": round(float(stress_arr.std()), 4),
                "stress_max": round(float(stress_arr.max()), 4),
                "confidence_mean": round(float(conf_arr.mean()), 4),
                "confidence_cnn": round(float(max(0.0, min(1.0, (conf_arr.mean() * 0.5 + 0.5)))), 4),
                "filler_rate": round(float(filler), 4),
                "phq8": {
                    "phq8_score": phq_res["phq8_score"],
                    "severity": phq_res["severity"],
                    "depressed_flag": phq_res["depressed_flag"],
                    "mean_stress": phq_res["mean_stress"],
                    "mean_confidence": phq_res["mean_confidence"],
                },
                "fsm_state_pct": fsm_pct,
                "fsm_state_series": fsm_series,
                "fsm_heatmap_100": resample_to_100(
                    [STATE_LEVELS.index(s) for s in fsm_series]
                ).tolist(),
                "stress_curve_100": resample_to_100(p_stress).tolist(),
                "confidence_curve_100": resample_to_100(p_conf).tolist(),
            }
            all_stress_curves.append(p_stress)
            all_conf_curves.append(p_conf)
            all_fsm_series.append(
                [STATE_LEVELS.index(s) for s in fsm_series]
            )
        except Exception as e:
            logger.error(f"Error processing {pid}: {e}")
            all_results[pid] = {"error": str(e)}

    stress_means = [r.get("stress_mean", 0) for r in all_results.values() if "stress_mean" in r]
    conf_means = [r.get("confidence_mean", 0) for r in all_results.values() if "confidence_mean" in r]
    phq_scores = [r.get("phq8", {}).get("phq8_score", 0) for r in all_results.values() if "phq8" in r]

    population = {
        "n_participants": len(all_results),
        "stress_mean": round(float(np.mean(stress_means)), 4) if stress_means else 0,
        "stress_std": round(float(np.std(stress_means)), 4) if stress_means else 0,
        "conf_mean": round(float(np.mean(conf_means)), 4) if conf_means else 0,
        "conf_std": round(float(np.std(conf_means)), 4) if conf_means else 0,
        "phq8_mean": round(float(np.mean(phq_scores)), 2) if phq_scores else 0,
        "phq8_elevated": sum(1 for s in phq_scores if s >= 10),
    }

    return {
        "n_participants": len(all_results),
        "participants": all_results,
        "population": population,
        "fsm_heatmap_100": np.stack(
            [resample_to_100(series) for series in all_fsm_series]
        ).tolist() if all_fsm_series else [],
        "participant_ids": [os.path.basename(p) for p in participants],
    }


def run_demo_session(domain: str = "general", duration: int = 60) -> Dict:
    """Run a simulated session with random realistic-ish metrics."""
    return run_offline_from_files(domain=domain)


def plot_summary(summary: Dict, output_dir: str = "."):
    """Generate graphs from session summary. Uses matplotlib if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("matplotlib not available, skipping graph generation")
        return []

    os.makedirs(output_dir, exist_ok=True)
    fig_paths = []

    # 1. Stress & Confidence over time
    fig, ax = plt.subplots(figsize=(12, 5))
    stress = summary.get("stress_series", [])
    conf = summary.get("confidence_series", [])
    t = summary.get("time_seconds", list(range(len(stress))))
    if stress:
        ax.plot(t[:len(stress)], [s * 100 for s in stress], color="#ef4444", linewidth=2, label="Stress (norm %)")
    if conf:
        ax.plot(t[:len(conf)], [c * 100 for c in conf], color="#3b82f6", linewidth=2, label="Confidence CNN %")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Score")
    ax.set_title("Stress & Confidence Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    p = os.path.join(output_dir, "stress_confidence_timeline.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    fig_paths.append(p)

    # 2. FSM state timeline
    fig, ax = plt.subplots(figsize=(12, 3))
    fsm_states = summary.get("fsm_state_series", [])
    if fsm_states:
        state_map = {"Baseline": 0, "Monitor": 1, "Adapt": 2, "Recover": 3, "Escalate": 4}
        state_nums = [state_map.get(s, 0) for s in fsm_states]
        t_fsm = list(range(len(state_nums)))
        ax.step(t_fsm, state_nums, where="post", linewidth=2, color="#8b5cf6")
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["Baseline", "Monitor", "Adapt", "Recover", "Escalate"])
        ax.set_xlabel("Time (seconds)")
        ax.set_title("FSM Adaptive Questioning State")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(output_dir, "fsm_state_timeline.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    fig_paths.append(p)

    # 3. FSM state distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    fsm_pct = summary.get("fsm_state_pct", summary.get("fsm_session", {}).get("pct_states", {}))
    if fsm_pct:
        labels = list(fsm_pct.keys())
        sizes = list(fsm_pct.values())
        colors = ["#3b82f6", "#8b5cf6", "#ef4444", "#22c55e", "#f97316"]
        ax.bar(labels, sizes, color=colors[: len(labels)])
        ax.set_ylabel("% of Session")
        ax.set_title("FSM State Distribution")
        ax.set_ylim(0, 100)
    fig.tight_layout()
    p = os.path.join(output_dir, "fsm_distribution.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    fig_paths.append(p)

    # 4. PHQ-8 score card
    fig, ax = plt.subplots(figsize=(4, 3))
    phq8 = summary.get("phq8", {})
    score = phq8.get("phq8_score", 0)
    severity = phq8.get("severity", "N/A")
    ax.text(0.5, 0.6, f"PHQ-8: {score:.1f}", ha="center", va="center", fontsize=28, fontweight="bold", color="#1e293b")
    ax.text(0.5, 0.3, f"Severity: {severity}", ha="center", va="center", fontsize=14, color="#64748b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    p = os.path.join(output_dir, "phq8_score.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    fig_paths.append(p)

    logger.info(f"Generated {len(fig_paths)} graph(s): {fig_paths}")
    return fig_paths


def plot_fsm_heatmap(agg_report: Dict, output_dir: str = ".") -> Optional[str]:
    """Generate a heatmap of FSM states across ALL participants.
    X-axis: normalized session progress (0-100%)
    Y-axis: participants
    Color: FSM state (Baseline/Monitor/Adapt/Recover/Escalate)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not available, skipping FSM heatmap")
        return None

    heat = agg_report.get("fsm_heatmap_100") or []
    pids = agg_report.get("participant_ids") or list(agg_report.get("participants", {}).keys())
    if not heat:
        logger.warning("No FSM heatmap data available")
        return None

    heat = np.array(heat, np.float32)  # (n_participants, 100), values in [0,4]
    n_p = heat.shape[0]
    pids = pids[:n_p]

    STATE_LEVELS = ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"]
    colors = ["#3b82f6", "#8b5cf6", "#ef4444", "#22c55e", "#f97316"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, max(3, n_p * 0.55)))
    im = ax.imshow(heat, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", extent=[0, 100, n_p, 0])

    ax.set_yticks(np.arange(n_p) + 0.5)
    ax.set_yticklabels(pids, fontsize=9)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Normalized Session Progress (%)")
    ax.set_ylabel("Participants")
    ax.set_title("FSM Adaptive Questioning State — All Participants (Heatmap)")

    legend = [Patch(facecolor=c, edgecolor="none", label=s)
              for c, s in zip(colors, STATE_LEVELS)]
    ax.legend(handles=legend, bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False, fontsize=9)

    fig.tight_layout()
    p = os.path.join(output_dir, "fsm_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated FSM heatmap: {p}")
    return p


def plot_phq8_scores(agg_report: Dict, output_dir: str = ".") -> Optional[str]:
    """Generate a bar chart of PHQ-8 scores across ALL participants,
    color-coded by severity, with an elevated-risk threshold line (score >= 10)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not available, skipping PHQ-8 plot")
        return None

    participants = agg_report.get("participants", {})

    rows = []
    for pid, data in participants.items():
        phq = data.get("phq8") or {}
        if "phq8_score" not in phq:
            continue
        rows.append((pid, phq.get("phq8_score", 0), phq.get("severity", "N/A")))

    if not rows:
        logger.warning("No per-participant PHQ-8 data available")
        return None

    rows.sort(key=lambda r: r[1])
    pids = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    sevs = [r[2] for r in rows]

    sev_color = {
        "None/Minimal": "#22c55e",
        "Mild": "#facc15",
        "Moderate": "#f97316",
        "Moderately Severe": "#ef4444",
        "Severe": "#7f1d1d",
    }
    bcolors = [sev_color.get(s, "#94a3b8") for s in sevs]

    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.5)))
    bars = ax.barh(pids, scores, color=bcolors, edgecolor="white")

    ax.axvline(10, color="#64748b", linestyle="--", linewidth=1.5,
               label="Elevated risk (score ≥ 10)")
    ax.set_xlim(0, 24)
    ax.set_xlabel("PHQ-8 Score")
    ax.set_ylabel("Participant")
    ax.set_title("PHQ-8 Depression Screening — All Participants")

    for bar, sc in zip(bars, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{sc:.1f}", va="center", fontsize=9, color="#1e293b")

    handles = [Patch(facecolor=sev_color[k], edgecolor="none", label=k)
               for k in ["None/Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"]
               if any(s == k for s in sevs)]
    handles.append(ax.get_legend_handles_labels()[0][0])
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False, fontsize=9)

    fig.tight_layout()
    p = os.path.join(output_dir, "phq8_scores.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Generated PHQ-8 scores plot: {p}")
    return p


def main():
    parser = argparse.ArgumentParser(description="Offline AI Interview Research Analysis")
    parser.add_argument("--data_dir", type=str, help="Path to participant directory (single *_P dir) or parent directory (all *_P dirs)")
    parser.add_argument("--demo", action="store_true", help="Run simulated session")
    parser.add_argument("--all", action="store_true", help="Analyze ALL participant directories under data_dir")
    parser.add_argument("--output", type=str, default="./analysis_output", help="Output directory")
    parser.add_argument("--domain", type=str, default="general", help="Interview domain")
    parser.add_argument("--duration", type=int, default=60, help="Demo session duration in seconds")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.demo or not args.data_dir:
        logger.info("Running demo session...")
        summary = run_demo_session(args.domain, args.duration)
    elif args.all:
        logger.info(f"Running analysis on all participants in {args.data_dir}...")
        summary = run_all_participants(args.data_dir, args.domain)

        report_path = os.path.join(args.output, "all_participants_report.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved aggregated report: {report_path}")

        plot_fsm_heatmap(summary, args.output)
        plot_phq8_scores(summary, args.output)

        print("\n" + "=" * 60)
        print("ALL PARTICIPANTS REPORT")
        print("=" * 60)
        for pid, data in summary.get("participants", {}).items():
            print(f"  {pid}: stress={data.get('stress_mean', 0):.4f}, "
                  f"conf={data.get('confidence_mean', 0):.4f}, "
                  f"phq8={data.get('phq8', {}).get('phq8_score', 0):.1f} "
                  f"({data.get('phq8', {}).get('severity', 'N/A')})")
        pop = summary.get("population", {})
        print(f"  Population: stress={pop.get('stress_mean', 0):.4f}, "
              f"conf={pop.get('conf_mean', 0):.4f}, "
              f"phq8 mean={pop.get('phq8_mean', 0):.1f}, "
              f"elevated={pop.get('phq8_elevated', 0)}/{pop.get('n_participants', 0)}")
        print("=" * 60)
        return
    else:
        base = args.data_dir.rstrip("/\\")
        if base.endswith("_P"):
            summary = run_offline_from_participant_dir(args.data_dir, args.domain)
        else:
            cov_files = glob.glob(os.path.join(args.data_dir, "*COVAREP*"))
            oface_files = glob.glob(os.path.join(args.data_dir, "*CLNF*AU*")) + glob.glob(os.path.join(args.data_dir, "*OpenFace_AU*"))
            transcript_files = glob.glob(os.path.join(args.data_dir, "*Transcript*"))
            summary = run_offline_from_files(
                covarep_path=cov_files[0] if cov_files else None,
                openface_path=oface_files[0] if oface_files else None,
                transcript_path=transcript_files[0] if transcript_files else None,
                domain=args.domain,
            )

    # Save report
    report_path = os.path.join(args.output, "report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved report: {report_path}")

    # Generate graphs
    fig_paths = plot_summary(summary, args.output)
    if fig_paths:
        logger.info(f"Graphs saved to: {fig_paths}")

    # Print summary
    print("\n" + "=" * 60)
    print("SESSION ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Participant:    {summary.get('participant_id', 'N/A')}")
    print(f"  Duration:        {summary.get('duration_seconds', 0):.1f}s")
    print(f"  Frames:          {summary.get('n_frames', 0)}")
    print(f"  Stress (mean):   {summary.get('stress_mean', 0):.4f}")
    print(f"  Stress (max):    {summary.get('stress_max', 0):.4f}")
    print(f"  Confidence:      {summary.get('confidence_mean', 0):.4f}")
    phq8 = summary.get("phq8", {})
    print(f"  PHQ-8:           {phq8.get('phq8_score', 0):.1f} ({phq8.get('severity', 'N/A')})")
    print(f"  Depressed flag:  {phq8.get('depressed_flag', False)}")
    print(f"  Pearson r (AU):  {summary.get('pearson_r_au_cnn', 0):.4f}")
    print(f"  Spikes:          {len(summary.get('bilstm_spikes', []))}")
    print(f"  Changepoints:    {len(summary.get('changepoints', []))}")
    fsm_s = summary.get("fsm_session", {})
    print(f"  FSM transitions: {fsm_s.get('n_transitions', 0)}")
    print(f"  FSM final state: {fsm_s.get('final_state', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
