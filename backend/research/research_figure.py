"""
Research Figure — Print-friendly 1400x900 composite for research papers.

Generates exactly four matplotlib panels arranged in a 2x2 grid, embedded
as base64 PNG images in a self-contained HTML page designed to be
screenshotted at 1400x900 pixels.
"""
import io
import base64
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger("research_figure")

# ---------------------------------------------------------------------------
# State color mappings
# ---------------------------------------------------------------------------
STATE_COLORS = {
    "Baseline": "#3b82f6",
    "Monitor": "#8b5cf6",
    "Adapt": "#ef4444",
    "Recover": "#22c55e",
    "Escalate": "#f97316",
}
STATE_ORDER = ["Baseline", "Monitor", "Adapt", "Recover", "Escalate"]
STATE_ID = {s: i for i, s in enumerate(STATE_ORDER)}

PHQ8_BANDS = [
    (0, 4, "#22c55e", "None/Min"),
    (4, 9, "#facc15", "Mild"),
    (9, 14, "#f97316", "Moderate"),
    (14, 19, "#ef4444", "Mod.Sev"),
    (19, 24, "#7f1d1d", "Severe"),
]


def _setup_rcparams():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return plt


def _load_summary(summary: Dict) -> Dict:
    """Extract all needed series from a summary dict with safe fallbacks."""
    per_second = summary.get("per_second", [])

    # Stress series
    stress = summary.get("stress_series", [])
    if not stress and per_second:
        stress = [r.get("stress_proxy_norm", 0) for r in per_second]

    # Confidence series
    conf = summary.get("confidence_series", [])
    if not conf and per_second:
        conf = [r.get("confidence_cnn", 0) for r in per_second]

    # Time series
    time_sec = summary.get("time_seconds", [])
    if not time_sec and stress:
        time_sec = list(range(len(stress)))

    # Convert Unix timestamps to elapsed seconds if needed
    if time_sec and max(time_sec) > 10000:
        t0 = time_sec[0]
        time_sec = [t - t0 for t in time_sec]

    # FSM state series
    fsm_series = summary.get("fsm_state_series", [])
    if not fsm_series and per_second:
        fsm_series = [r.get("fsm_state", "Baseline") for r in per_second]

    # Spike timestamps (from per_second spike_alert flags)
    spike_timestamps = []
    for i, r in enumerate(per_second):
        if r.get("spike_alert"):
            if i < len(time_sec):
                spike_timestamps.append(time_sec[i])
    # Also check bilstm_spikes
    for sp in summary.get("bilstm_spikes", []):
        if isinstance(sp, dict) and "timestamp" in sp:
            t = sp["timestamp"]
            if t is not None:
                # Convert Unix timestamp to elapsed if needed
                if time_sec and max(time_sec) > 10000 and t > 10000:
                    t = t - time_sec[0]
                if t not in spike_timestamps:
                    spike_timestamps.append(float(t))
    # Also check stress_spikes
    for sp in summary.get("stress_spikes", []):
        if isinstance(sp, dict):
            idx = sp.get("step_index")
            if isinstance(idx, (int, float)) and int(idx) < len(time_sec):
                spike_timestamps.append(time_sec[int(idx)])
    spike_timestamps.sort()

    # Changepoints (list of indices)
    changepoints = summary.get("changepoints", [])
    changepoint_times = []
    for cp in changepoints:
        if isinstance(cp, (int, float)):
            idx = int(cp)
            if idx < len(time_sec):
                changepoint_times.append(time_sec[idx])

    # FSM state percentages
    pct_states = summary.get("fsm_state_pct", {})
    if not pct_states:
        pct_states = (summary.get("fsm_session") or {}).get("pct_states", {})

    # FSM session summary
    fsm_session = summary.get("fsm_session", {})
    n_transitions = fsm_session.get("n_transitions", 0)
    final_state = fsm_session.get("final_state", "Baseline")

    # PHQ-8
    phq8 = summary.get("phq8", {})
    phq8_score = phq8.get("phq8_score", summary.get("phq8_score", 0.0))
    if phq8_score is None:
        phq8_score = 0.0
    severity = phq8.get("severity", "N/A")
    depressed_flag = phq8.get("depressed_flag", False)
    depressed_label = "Depressed" if depressed_flag else "Not depressed"

    # Means
    stress_mean = summary.get("stress_mean", 0.0)
    if stress_mean is None:
        stress_mean = 0.0
    confidence_mean = summary.get("confidence_mean", 0.0)
    if confidence_mean is None:
        confidence_mean = 0.0

    # Session metadata
    sid = summary.get("session_id", "unknown")
    domain = summary.get("domain", "general")
    duration = summary.get("duration_seconds", 0.0)
    if duration is None:
        duration = 0.0

    # Number of questions answered — check multiple sources.
    # Prefer explicit counts; NEVER fall back to len(per_second) (that is
    # the per-second frame count, not the number of questions asked).
    n_questions = (
        summary.get("n_questions")
        or summary.get("question_count")
        or summary.get("questions_answered")
        or (summary.get("fsm_session") or {}).get("question_count")
        or len(summary.get("detail_per_question", []))
    )

    # Ensure all lists are same length for safe indexing
    n = max(len(stress), len(conf), len(time_sec), len(fsm_series))
    if n == 0:
        n = 1  # avoid empty

    def _pad(lst, length, default=0.0):
        while len(lst) < length:
            lst.append(default)
        return lst

    stress = _pad(stress, n)
    conf = _pad(conf, n)
    time_sec = _pad(time_sec, n, default=0.0)
    fsm_series = _pad(fsm_series, n, default="Baseline")

    return {
        "stress": stress,
        "conf": conf,
        "time_seconds": time_sec,
        "fsm_series": fsm_series,
        "spike_timestamps": spike_timestamps,
        "changepoint_times": changepoint_times,
        "pct_states": pct_states,
        "n_transitions": n_transitions,
        "final_state": final_state,
        "phq8_score": float(phq8_score),
        "severity": severity,
        "depressed_flag": depressed_flag,
        "depressed_label": depressed_label,
        "stress_mean": float(stress_mean),
        "confidence_mean": float(confidence_mean),
        "session_id": sid,
        "domain": domain,
        "duration": float(duration),
        "n_questions": n_questions,
    }


# ---------------------------------------------------------------------------
# PANEL 1 — BiLSTM Temporal Stress & AU-CNN Confidence Timeline
# ---------------------------------------------------------------------------
def _panel1_timeline(plt, data: Dict) -> bytes:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    time_seconds = list(data["time_seconds"])
    stress_series = list(data["stress"])
    confidence_series = list(data["conf"])

    # --- X-axis elapsed time fix ---
    # Detect and convert Unix timestamps to elapsed seconds
    if time_seconds and len(time_seconds) > 0:
        first = time_seconds[0]

        # Case 1: Raw Unix timestamps (values > 1,000,000,000)
        if isinstance(first, (int, float)) and first > 1_000_000_000:
            t0_raw = first
            time_seconds = [t - t0_raw for t in time_seconds]

        # Case 2: ISO datetime strings
        elif isinstance(first, str):
            from datetime import datetime
            try:
                parsed = [datetime.fromisoformat(t) for t in time_seconds]
                t0_raw = parsed[0]
                time_seconds = [(t - t0_raw).total_seconds() for t in parsed]
            except Exception:
                t0_raw = 0
                time_seconds = list(range(len(time_seconds)))

        # Case 3: Already elapsed seconds (values < 10,000) — use as-is
        else:
            t0_raw = 0
    else:
        # Fallback: generate 0,1,2,... from length of stress series
        t0_raw = 0
        time_seconds = list(range(len(stress_series)))

    # Ensure time and data series are the same length
    min_len = min(len(time_seconds), len(stress_series),
                  len(confidence_series))
    time_seconds = time_seconds[:min_len]
    stress_plot = np.array(stress_series[:min_len]) * 100
    conf_plot = np.array(confidence_series[:min_len]) * 100
    # --- End fix ---

    ax.plot(time_seconds, stress_plot, color="#ef4444", linewidth=1.8,
            label="Stress proxy (Eq. 3.1)")
    ax.plot(time_seconds, conf_plot, color="#3b82f6", linewidth=1.8,
            label="AU-CNN confidence (Eq. 3.3)")

    # Spike vertical lines — convert to elapsed seconds
    for spike in data["spike_timestamps"]:
        spike_t = spike.get("timestamp", 0) if isinstance(spike, dict) else spike
        if spike_t > 1_000_000_000:
            spike_t = spike_t - t0_raw
        if 0 <= spike_t <= time_seconds[-1]:
            ax.axvline(x=spike_t, color="#ef4444", linestyle="--",
                       alpha=0.5, linewidth=1.2)

    # Changepoint vertical lines — convert to elapsed seconds
    for cp in data["changepoint_times"]:
        cp_val = cp.get("timestamp", 0) if isinstance(cp, dict) else cp
        if isinstance(cp_val, float) and cp_val > 1_000_000_000:
            cp_val = cp_val - t0_raw
        if 0 <= cp_val <= time_seconds[-1]:
            ax.axvline(x=cp_val, color="#f97316", linestyle="--",
                       alpha=0.5, linewidth=1.2)

    # Additional X-axis formatting — prevent scientific notation
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    ax.set_xlim(left=0)
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=9)
    ax.set_title("Temporal Stress & Confidence Timeline", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotation
    stress_mean = data["stress_mean"]
    conf_mean = data["confidence_mean"]
    n_spikes = len(data["spike_timestamps"])
    n_cp = len(data["changepoint_times"])
    ax.annotate(
        f"Mean stress: {stress_mean:.1%}  |  "
        f"Mean conf: {conf_mean:.1%}  |  "
        f"Spikes: {n_spikes}  |  Changepoints: {n_cp}",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=7, color="#444444",
        ha="left", va="bottom",
    )

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PANEL 2 — FSM State Timeline
# ---------------------------------------------------------------------------
def _panel2_fsm_timeline(plt, data: Dict) -> bytes:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    t = data["time_seconds"]
    fsm = data["fsm_series"]
    n = len(fsm)

    # Map FSM states to numeric IDs for step plot
    fsm_ids = [STATE_ID.get(s, 0) for s in fsm]
    t_arr = np.array(t[:n])

    # Step chart with fill
    ax.step(t_arr, fsm_ids, where="post", color="#333333", linewidth=0.8)

    # Fill between with state colors
    for state_name, sid_val in STATE_ID.items():
        color = STATE_COLORS[state_name]
        # Create mask for this state
        mask = np.array(fsm_ids) == sid_val
        # Fill segments where this state is active
        for i in range(n - 1):
            if mask[i]:
                ax.fill_between(
                    [t_arr[i], t_arr[i + 1]],
                    [sid_val, sid_val],
                    [0, 0],
                    color=color, alpha=0.25, step="post",
                )
        # Also fill last point
        if mask[n - 1] and n > 0:
            ax.fill_between(
                [t_arr[n - 1], t_arr[n - 1] + 1],
                [sid_val, sid_val],
                [0, 0],
                color=color, alpha=0.25, step="post",
            )

    # FSM transition markers
    fsm_session = {}
    for sp in data.get("fsm_series", []):
        pass
    # Use transition times from fsm_session data if available
    # We already have n_transitions; derive from state changes
    prev = None
    for i, s in enumerate(fsm):
        if prev is not None and s != prev and i < len(t):
            ax.axvline(t[i], color="#999999", linestyle=":", linewidth=0.8,
                       alpha=0.7)
        prev = s

    ax.set_yticks(range(len(STATE_ORDER)))
    ax.set_yticklabels(STATE_ORDER, fontsize=8)
    ax.set_ylim(-0.5, len(STATE_ORDER) - 0.5)
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_title("FSM Adaptive State Timeline (Eq. 3.4)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    # Annotation
    pct = data["pct_states"]
    pct_baseline = pct.get("Baseline", 0.0)
    pct_adapt = pct.get("Adapt", 0.0)
    ax.annotate(
        f"Transitions: {data['n_transitions']}  |  "
        f"Final state: {data['final_state']}  |  "
        f"Baseline: {pct_baseline:.1f}%  |  "
        f"Adapt: {pct_adapt:.1f}%",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=7, color="#444444",
        ha="left", va="bottom",
    )

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PANEL 3 — FSM State Distribution Bar Chart
# ---------------------------------------------------------------------------
def _panel3_fsm_bars(plt, data: Dict) -> bytes:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    pct = data["pct_states"]

    vals = [pct.get(s, 0.0) for s in STATE_ORDER]
    colors = [STATE_COLORS[s] for s in STATE_ORDER]

    bars = ax.bar(STATE_ORDER, vals, color=colors, width=0.5)

    # Value labels on top
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # Reference lines
    ax.axhline(47.95, color="#666666", linestyle="--", linewidth=1,
               label="Population Baseline mean (47.95%)")
    ax.axhline(24.08, color="#ef4444", linestyle="--", linewidth=1,
               label="Population Adapt mean (24.08%)")

    ax.set_ylabel("% of Session Time", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("FSM State Distribution vs. Population (Eq. 3.4)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PANEL 4 — PHQ-8 Proxy Scorecard
# ---------------------------------------------------------------------------
def _panel4_phq8(plt, data: Dict) -> bytes:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.axis("off")

    score = data["phq8_score"]
    severity = data["severity"]
    depressed_label = data["depressed_label"]
    stress_mean = data["stress_mean"]
    conf_mean = data["confidence_mean"]

    # --- Severity gauge bar ---
    gauge_left = 0.05
    gauge_right = 0.95
    gauge_width = gauge_right - gauge_left
    gauge_y = 0.42
    gauge_h = 0.10

    band_colors = ["#22c55e", "#facc15", "#f97316", "#ef4444", "#7f1d1d"]
    band_labels = ["None/Min", "Mild", "Moderate", "Mod.Sev", "Severe"]

    for i, (lo, hi, color, label) in enumerate(PHQ8_BANDS):
        x_left = gauge_left + (lo / 24) * gauge_width
        x_right = gauge_left + (hi / 24) * gauge_width
        rect = plt.Rectangle(
            (x_left, gauge_y), x_right - x_left, gauge_h,
            facecolor=color, edgecolor="white", linewidth=1,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        # Label in middle of band
        x_mid = (x_left + x_right) / 2
        ax.text(x_mid, gauge_y - 0.02, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=6, color="#374151")

    # --- Triangle marker ---
    marker_x = gauge_left + (min(24, max(0, score)) / 24) * gauge_width
    ax.plot(marker_x, 0.54, marker="v", color="black", markersize=12,
            transform=ax.transAxes, clip_on=False)

    # --- Large score text ---
    # Find the band color for this score
    band_color = "#374151"
    for lo, hi, color, _ in PHQ8_BANDS:
        if lo <= score <= hi:
            band_color = color
            break
    ax.text(0.5, 0.78, f"{score:.1f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=42, fontweight="bold",
            color=band_color)

    # --- Severity + depressed label ---
    ax.text(0.5, 0.67, f"{severity}  \u00b7  {depressed_label}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="#374151")

    # --- Mean stress/confidence ---
    ax.text(0.5, 0.28,
            f"Mean Stress: {stress_mean:.1%}   Mean Confidence: {conf_mean:.1%}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#6b7280")

    # --- Disclaimer ---
    ax.text(0.5, 0.12,
            "Research screening proxy only \u2014 not a clinical diagnosis.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color="#9ca3af", style="italic")

    # --- Title ---
    ax.text(0.5, 0.95, "PHQ-8 Depression Screening Proxy (Eqs. 5.1\u20135.3)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="black")

    # --- Thin border ---
    for spine in ax.spines.values():
        spine.set_edgecolor("#e5e7eb")
        spine.set_linewidth(1)
        spine.set_visible(True)

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# HTML page builder
# ---------------------------------------------------------------------------
def _build_html(
    img1_b64: bytes,
    img2_b64: bytes,
    img3_b64: bytes,
    img4_b64: bytes,
    session_id: str,
    domain: str,
    duration: float,
    n_questions: int,
) -> str:
    sid_short = session_id[:8] if len(session_id) >= 8 else session_id
    q_str = f"{n_questions}" if n_questions else "N/A"
    title = (
        f"InterviewIQ \u2014 Single-Session Research Analysis  |  "
        f"Session: {sid_short}...  |  Domain: {domain}  |  "
        f"Duration: {duration:.0f}s  |  {q_str} questions"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1400, initial-scale=1">
<title>Research Figure — {sid_short}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html {{ width:1400px; height:900px; overflow:hidden; }}
  body {{ width:1400px; height:900px; overflow:hidden; background:#ffffff;
         font-family:monospace; padding:10px 12px 10px 12px;
         display:flex; flex-direction:column; }}
  .title {{ font-size:11px; color:#6b7280; margin-bottom:6px; flex-shrink:0;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .grid {{ display:grid;
           grid-template-columns:1fr 1fr;
           grid-template-rows:1fr 1fr;
           gap:6px;
           flex:1;
           min-height:0; }}
  .cell {{ border:1px solid #e5e7eb; border-radius:6px;
           overflow:hidden; background:white; min-height:0; }}
  .cell img {{ width:100%; height:100%; object-fit:contain; display:block; }}
</style>
</head>
<body>
  <div class="title">{title}</div>
  <div class="grid">
    <div class="cell"><img src="data:image/png;base64,{img1_b64}"></div>
    <div class="cell"><img src="data:image/png;base64,{img2_b64}"></div>
    <div class="cell"><img src="data:image/png;base64,{img3_b64}"></div>
    <div class="cell"><img src="data:image/png;base64,{img4_b64}"></div>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_research_figure(summary: Dict) -> str:
    """Generate the four-panel research figure and return an HTML string.

    Parameters
    ----------
    summary : dict
        The finalized InterviewEngine summary dict (from ``finalize()``).

    Returns
    -------
    str
        Self-contained HTML page with the four embedded base64 PNG images.
    """
    plt = _setup_rcparams()
    data = _load_summary(summary)

    img1 = _panel1_timeline(plt, data)
    img2 = _panel2_fsm_timeline(plt, data)
    img3 = _panel3_fsm_bars(plt, data)
    img4 = _panel4_phq8(plt, data)

    b64_1 = base64.b64encode(img1).decode("ascii")
    b64_2 = base64.b64encode(img2).decode("ascii")
    b64_3 = base64.b64encode(img3).decode("ascii")
    b64_4 = base64.b64encode(img4).decode("ascii")

    html = _build_html(
        b64_1, b64_2, b64_3, b64_4,
        session_id=data["session_id"],
        domain=data["domain"],
        duration=data["duration"],
        n_questions=data["n_questions"],
    )
    logger.info(
        f"Generated research figure for session {data['session_id'][:8]} "
        f"({len(img1)+len(img2)+len(img3)+len(img4)} bytes total PNG)"
    )
    return html
