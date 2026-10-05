"""
E-DAIC → DAIC-WOZ Preprocessing Adapter (Mandal et al., 2024)

Maps Extended-DAIC (E-DAIC) file formats to DAIC-WOZ compatible layout:
  - OpenSMILE eGeMAPS (88 cols) → COVAREP (74 cols) via zero-fill mapping
  - OpenFace 2.1 CLNF_AUs.txt → DAIC-WOZ naming convention
  - Transcript CSV preserved as-is

Also provides batch readers for running the research pipeline offline.
"""
import os
import csv
import json
import glob
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("data_adapter")


# ---- eGeMAPS → COVAREP column mapping ---- #
# eGeMAPS has 88 features; COVAREP has 74.
# We map the first 74 overlapping/direct columns, zero-fill the rest.
EGEMAPS_TO_COVAREP_MAP = {
    "F0semitoneFrom27.5Hz_sma3nz": 0,
    "F0semitoneFrom27.5Hz_sma3nz_pctl2": 1,
    "F0semitoneFrom27.5Hz_sma3nz_pctl5": 2,
    "F0semitoneFrom27.5Hz_sma3nz_pctl8": 3,
    "loudness_sma3": 4,
    "loudness_sma3_pctl2": 5,
    "loudness_sma3_pctl5": 6,
    "loudness_sma3_pctl8": 7,
    "jitterLocal_sma3nz": 8,
    "shimmerLocal_sma3nz": 9,
    "HNRdBACF_sma3nz": 10,
    "logRelF0-H1-H2_sma3nz": 11,
    "logRelF0-H1-A_sma3nz": 12,
    "F1frequency_sma3nz": 13,
    "F1bandwidth_sma3nz": 14,
    "F1amplitudeLogRelF0_sma3nz": 15,
    "F2frequency_sma3nz": 16,
    "F2bandwidth_sma3nz": 17,
    "F2amplitudeLogRelF0_sma3nz": 18,
    "F3frequency_sma3nz": 19,
    "F3bandwidth_sma3nz": 20,
    "F3amplitudeLogRelF0_sma3nz": 21,
    "F4frequency_sma3nz": 22,
    "F4bandwidth_sma3nz": 23,
    "F4amplitudeLogRelF0_sma3nz": 24,
    "MMRacs33_sma3nz": 25,
    "avgMelspectrogram1_sma3": 26,
    "avgMelspectrogram2_sma3": 27,
    "avgMelspectrogram3_sma3": 28,
    "avgMelspectrogram4_sma3": 29,
    "avgMelspectrogram5_sma3": 30,
    "avgMelspectrogram6_sma3": 31,
    "avgMelspectrogram7_sma3": 32,
    "avgMelspectrogram8_sma3": 33,
    "avgMelspectrogram9_sma3": 34,
    "avgMelspectrogram10_sma3": 35,
    "avgMelspectrogram11_sma3": 36,
    "avgMelspectrogram12_sma3": 37,
    "spectralFlux_sma3": 38,
    "MFCC1_sma3nz": 39,
    "MFCC2_sma3nz": 40,
    "MFCC3_sma3nz": 41,
    "MFCC4_sma3nz": 42,
    "logF0frequency_sma3nz": 43,
    "jitterDDP_sma3nz": 44,
    "shimmerAPQ3_sma3nz": 45,
    "shimmerAPQ5_sma3nz": 46,
    "shimmerDDA_sma3nz": 47,
    "equalLoudness_sma3": 48,
    "hammarIndex_sma3nz": 49,
    "loudnessPeaksRatio_sma3": 50,
    "meanLoudness_sma3": 51,
    "loudness_sma3percentile20": 52,
    "loudness_sma3percentile50": 53,
    "loudness_sma3percentile80": 54,
    "loudness_sma3meanNonzero": 55,
    "loudness_sma3meanRange": 56,
    "F0semitoneFrom27.5Hz_sma3meanNonzero": 57,
    "F0semitoneFrom27.5Hz_sma3meanRange": 58,
}


def egemaps_csv_to_covarep(csv_path: str) -> Optional[np.ndarray]:
    """Read an eGeMAPS CSV and map to 74-col COVAREP layout."""
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        n_frames = len(rows)
        cov = np.zeros((n_frames, 74), np.float32)
        for i, row in enumerate(rows):
            for eg_col, cov_col in EGEMAPS_TO_COVAREP_MAP.items():
                try:
                    cov[i, cov_col] = float(row.get(eg_col, 0))
                except (ValueError, KeyError):
                    pass
        return cov
    except Exception as e:
        logger.error(f"Failed to map eGeMAPS to COVAREP: {e}")
        return None


def adapt_edaic_to_daicwoz(edaic_dir: str, out_dir: str) -> Dict:
    """Adapt an E-DAIC participant directory to DAIC-WOZ compatible layout.

    Reads:
        <edaic_dir>/features/<ID>_OpenSMILE2.3.0_egemaps.csv
        <edaic_dir>/features/<ID>_OpenFace2.1.0_Pose_gaze_AUs.csv
        <edaic_dir>/<ID>_Transcript.csv

    Writes:
        <out_dir>/<ID>_P/<ID>_COVAREP.csv
        <out_dir>/<ID>_P/<ID>_CLNF_AUs.txt
        <out_dir>/<ID>_P/<ID>_Transcript.csv
    """
    base = os.path.basename(edaic_dir.rstrip("/\\"))
    pid = base.replace("_P", "")
    features_dir = os.path.join(edaic_dir, "features")

    participant_out = os.path.join(out_dir, f"{pid}_P")
    os.makedirs(participant_out, exist_ok=True)

    # eGeMAPS → COVAREP
    egemaps_file = None
    for f in glob.glob(os.path.join(features_dir, "*egemaps*")):
        egemaps_file = f
        break
    if egemaps_file:
        cov = egemaps_csv_to_covarep(egemaps_file)
        if cov is not None:
            out_path = os.path.join(participant_out, f"{pid}_COVAREP.csv")
            header = ["Frame"] + [f"F{i}" for i in range(74)]
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, row in enumerate(cov):
                    writer.writerow([i] + row.tolist())
            logger.info(f"Wrote COVAREP: {out_path} ({len(cov)} frames)")

    # OpenFace rename
    oface_file = None
    for f in glob.glob(os.path.join(features_dir, "*OpenFace*")):
        oface_file = f
        break
    if oface_file:
        out_path = os.path.join(participant_out, f"{pid}_CLNF_AUs.txt")
        with open(oface_file, "r") as fin, open(out_path, "w") as fout:
            for line in fin:
                fout.write(line)
        logger.info(f"Wrote OpenFace: {out_path}")

    # Transcript (copy as-is)
    for f in glob.glob(os.path.join(edaic_dir, "*Transcript*")):
        out_path = os.path.join(participant_out, f"{pid}_Transcript.csv")
        with open(f, "r") as fin, open(out_path, "w") as fout:
            for line in fin:
                fout.write(line)
        logger.info(f"Wrote Transcript: {out_path}")

    return {"pid": pid, "output_dir": participant_out}
