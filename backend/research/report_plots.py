"""
Report Plots — Server-side matplotlib PNG generation (Section 5) for a single
interview session, rendered from the session analyzer's results dict.

Produces exactly four plots:
  1. stress_confidence_timeline.png — stress & confidence over time with
     stress spikes, FSM transition markers, and changepoint segments (binseg).
  2. au_cnn_confidence.png — AU-CNN dual confidence (proxy vs CNN) plus a
     stress-vs-confidence scatter colored by FSM state.
  3. fsm_heatmap.png — single-participant FSM state heatmap (band over time)
     with a per-state % bar alongside.
  4. phq8_card.png — PHQ-8 score card with the spec gauge bands.

All plots are rendered server-side to PNG. Never client-side JS.
"""
import os
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger("report_plots")

STATE_COLORS = {
    "Baseline": "#3b82f6",
    "Monitor": "#8b5cf6",
    "Adapt": "#ef4444",
    "Recover": "#22c55e",
    "Escalate": "#f97316",
}
STATE_ORDER = ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"]

PHQ8_BANDS = [
    (0, 4, "#22c55e", "None/Minimal"),
    (4, 9, "#facc15", "Mild"),
    (9, 14, "#f97316", "Moderate"),
    (14, 19, "#ef4444", "Moderately Severe"),
    (19, 24, "#7f1d1d", "Severe"),
]


def _mp():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _series(summary: Dict) -> Dict:
    """Gather x/stress/conf series from the summary, using stress_curve_100 if
    present for a smoother viewed timeline, else the raw per-second series."""
    t = summary.get("time_seconds") or list(range(len(summary.get("stress_series", []))))
    stress = summary.get("stress_100") or summary.get("stress_series", [])
    conf = summary.get("confidence_100") or summary.get("confidence_series", [])
    if not stress and summary.get("per_second"):
        stress = [r.get("stress_proxy_norm", 0) for r in summary["per_second"]]
        conf = [r.get("confidence_proxy", 0) for r in summary["per_second"]]
        t = list(range(len(stress)))
    if stress is not None and t is None:
        t = list(range(len(stress)))
    return {"t": t, "stress": stress, "conf": conf}


def _transition_times(summary: Dict) -> List[int]:
    transitions = (summary.get("fsm_session") or {}).get("transition_log", [])
    # elapsed is in seconds since interview start; convert to step index
    times = []
    for tr in transitions:
        el = tr.get("elapsed")
        if isinstance(el, (int, float)):
            times.append(int(round(el)))
    return times


def plot_1_timeline(summary: Dict, out: str) -> str:
    plt = _mp()
    fig, ax = plt.subplots(figsize=(12, 5))
    d = _series(summary)
    t = d["t"]
    stress = d["stress"]
    conf = d["conf"]

    if stress:
        ax.plot(t[: len(stress)], np.asarray(stress) * 100, color="#ef4444",
                linewidth=2, label="Stress (norm %)")
    if conf:
        ax.plot(t[: len(conf)], np.asarray(conf) * 100, color="#3b82f6",
                linewidth=2, label="Confidence (AU-CNN %)")

    # Stress spikes (positions derived from per-second spike_alert flags)
    per_sec = summary.get("per_second", [])
    spike_indices = [i for i, r in enumerate(per_sec) if r.get("spike_alert")]
    if not spike_indices:
        for sp in summary.get("stress_spikes", []):
            idx = sp.get("step_index") if isinstance(sp, dict) else sp
            if isinstance(idx, (int, float)):
                spike_indices.append(int(idx))
    for idx in spike_indices:
        if 0 <= idx < len(t):
            ax.axvspan(idx - 0.5, idx + 0.5, color="#ef4444", alpha=0.18)

    # FSM transition markers
    for ti in _transition_times(summary):
        if 0 <= ti < len(t):
            ax.axvline(t[ti], color="#8b5cf6", linestyle="--", linewidth=1.2, alpha=0.8)

    # Changepoint segments (binseg) via step-shading between changepoint indices
    cps = summary.get("changepoints", [])
    if cps and stress:
        arr = np.asarray(stress)
        pts = [0] + [int(c) for c in cps if isinstance(c, (int, float))] + [len(arr)]
        pts = sorted(set(pts))
        for i in range(len(pts) - 1):
            if pts[i] < len(t):
                ax.axvspan(t[pts[i]], t[min(pts[i + 1], len(t) - 1)],
                           alpha=0.06, color="#f97316")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Stress & Confidence Over Time (with spikes / transitions / changepoints)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_2_au_cnn(summary: Dict, out: str) -> str:
    plt = _mp()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    per_sec = summary.get("per_second", [])
    t = list(range(len(per_sec))) if per_sec else list(
        range(len(summary.get("confidence_series", []))))
    proxy = [r.get("confidence_proxy", 0) * 100 for r in per_sec] if per_sec else []
    cnn = [r.get("confidence_cnn", 0) * 100 for r in per_sec] if per_sec else []

    ax = axes[0]
    if proxy:
        ax.plot(t[: len(proxy)], proxy, color="#a78bfa", linewidth=1.5,
                alpha=0.7, label="Confidence Proxy")
    if cnn:
        ax.plot(t[: len(cnn)], cnn, color="#3b82f6", linewidth=2, label="Confidence CNN")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Confidence (%)")
    ax.set_title("AU-CNN Dual Confidence (proxy vs CNN)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Stress vs Confidence scatter colored by FSM state
    ax = axes[1]
    states = [r.get("fsm_state", "Baseline") for r in per_sec] if per_sec else []
    stress_vals = [r.get("stress_pct", r.get("stress_proxy_norm", 0) * 100) for r in per_sec]
    conf_vals = [r.get("confidence_pct", r.get("confidence_cnn", 0) * 100) for r in per_sec]
    for st in STATE_ORDER:
        xs = [c for s, c in zip(states, conf_vals) if s == st]
        ys = [s for s_, s in zip(states, stress_vals) if s_ == st]
        if xs:
            ax.scatter(xs, ys, c=STATE_COLORS.get(st, "#888888"), alpha=0.6,
                       s=18, label=st)
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Stress (%)")
    ax.set_title("Stress vs Confidence (by FSM state)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_3_fsm_heatmap(summary: Dict, out: str) -> str:
    plt = _mp()
    fsm_series = (summary.get("fsm_state_series")
                  or [r.get("fsm_state", "Baseline") for r in summary.get("per_second", [])])
    pct = (summary.get("fsm_state_pct")
           or (summary.get("fsm_session") or {}).get("pct_states", {}))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [3, 1.2]})

    ax = axes[0]
    if fsm_series:
        n = len(fsm_series)
        state_id = {s: i for i, s in enumerate(STATE_ORDER)}
        rows = []
        seen = set()
        for s in fsm_series:
            if s not in seen:
                seen.add(s)
                rows.append((state_id.get(s, 0), s))
        # Actually build per-timestep state index
        arr = np.array([[state_id.get(s, 0)] for s in fsm_series], dtype=float).T
        cmap = plt.matplotlib.colors.ListedColormap(
            [STATE_COLORS.get(s, "#888888") for s in STATE_ORDER])
        im = ax.imshow(arr, aspect="auto", cmap=cmap, extent=[0, n, 0, 1], origin="lower")
        ax.set_yticks([])
        ax.set_xlabel("Time (seconds)")
        ax.set_title("FSM State Heatmap (per second)")
        # legend
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=STATE_COLORS[s], label=s) for s in STATE_ORDER]
        ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=1)
    else:
        ax.text(0.5, 0.5, "No FSM data", ha="center", va="center")

    ax = axes[1]
    if pct:
        labels = [s for s in STATE_ORDER if pct.get(s, 0) > 0]
        vals = [pct.get(s, 0) for s in labels]
        colors = [STATE_COLORS[s] for s in labels]
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        ax.set_xlabel("% of Session")
        ax.set_title("State Distribution")
        ax.set_xlim(0, 100)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _gauge_phq8(fig, ax, score: float):
    import matplotlib.patches as mpatches
    from matplotlib.patheffects import withStroke
    import matplotlib.pyplot as plt
    n_bands = 5
    width = 0.5
    angles = [180 + (i / n_bands) * 180 for i in range(n_bands + 1)]
    for i, (lo, hi, color, label) in enumerate(PHQ8_BANDS):
        theta0 = np.radians(180 + (lo / 24) * 180)
        theta1 = np.radians(180 + (hi / 24) * 180)
        mid = (theta0 + theta1) / 2
        outer, inner = 1.0, 1.0 - width
        verts = [
            (inner * np.cos(theta0), inner * np.sin(theta0)),
            (outer * np.cos(theta0), outer * np.sin(theta0)),
            (outer * np.cos(theta1), outer * np.sin(theta1)),
            (inner * np.cos(theta1), inner * np.sin(theta1)),
        ]
        poly = mpatches.Polygon(verts, closed=True, facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(poly)
        tick = np.cos((theta0 + theta1) / 2), np.sin((theta0 + theta1) / 2)
        ax.text(tick[0] * 1.06, tick[1] * 1.06, str(hi), ha="center", va="center", fontsize=8)

    # needle
    a = np.radians(180 + (min(24, max(0, score)) / 24) * 180)
    ax.plot([0, 0.8 * np.cos(a)], [0, 0.8 * np.sin(a)], color="#111827", linewidth=3,
            path_effects=[withStroke(linewidth=5, foreground="white")])
    ax.add_patch(plt.Circle((0, 0), 0.12, color="#111827", zorder=5))
    ax.text(0, 0.55, f"{score:.0f}", ha="center", va="center", fontsize=26, fontweight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.2, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_4_phq8(summary: Dict, out: str) -> str:
    plt = _mp()
    phq8 = summary.get("phq8") or {}
    score = phq8.get("phq8_score",
                     summary.get("phq8_score", 0.0))
    severity = phq8.get("severity", "N/A")
    n_frames = phq8.get("n_frames", summary.get("n_frames", 0))

    is_proxy = True  # PHQ-8 is always a computational proxy here (not clinical)

    fig, ax = plt.subplots(figsize=(6, 5))
    _gauge_phq8(fig, ax, float(score))
    ax.set_title("PHQ-8 Estimation (computational proxy)", fontsize=13)

    fig.text(0.5, 0.02, f"Score: {score:.1f}  |  Severity: {severity}  |  Frames: {n_frames}",
             ha="center", fontsize=10)
    fig.text(0.5, 0.085, "For research/evaluation only — NOT a clinical diagnosis.",
             ha="center", fontsize=8, color="#6b7280")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_report_plots(summary: Dict, output_dir: str) -> Dict[str, str]:
    """Generate all four session plots. Returns {key: absolute_path}."""
    os.makedirs(output_dir, exist_ok=True)
    plots = {
        "stress_confidence_timeline": plot_1_timeline(summary, os.path.join(output_dir, "stress_confidence_timeline.png")),
        "au_cnn_confidence": plot_2_au_cnn(summary, os.path.join(output_dir, "au_cnn_confidence.png")),
        "fsm_heatmap": plot_3_fsm_heatmap(summary, os.path.join(output_dir, "fsm_heatmap.png")),
        "phq8_card": plot_4_phq8(summary, os.path.join(output_dir, "phq8_card.png")),
    }
    logger.info(f"Generated {len(plots)} session plots in {output_dir}")
    return plots
