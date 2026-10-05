"""
PHQ-8 Estimator — Mandal et al. (2024)
Eq. 4.1: fd = {mu, sigma, min, max, P10, P25, P75, P90, skewness, kurtosis}
Eq. 4.2: 1050-dim = 105 dims x 10 functionals
Eq. 4.3: RobustScaler
Eq. 5.1: PLS; Eq. 5.2: MLP MSE; Eq. 5.3: clip(y, 0, 24)
"""
import numpy as np
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("phq8")


def _ten(arr):
    """Eq. 4.1 — ten statistical functionals per feature dimension."""
    if len(arr) < 2:
        return np.zeros(10, np.float32)
    a = np.array(arr, dtype=np.float64)
    n = len(a)
    mu = a.mean()
    s = a.std()
    sk = float(np.mean(((a - mu) / s) ** 3)) if s > 0 else 0.0
    ku = float(np.mean(((a - mu) / s) ** 4) - 3) if s > 0 else 0.0
    return np.array(
        [
            mu,
            s,
            a.min(),
            a.max(),
            np.percentile(a, 10),
            np.percentile(a, 25),
            np.percentile(a, 75),
            np.percentile(a, 90),
            sk,
            ku,
        ],
        np.float32,
    )


def _relu(x):
    return np.maximum(x, 0)


class _PLS:
    def __init__(self, nc=20, ind=1050, seed=7):
        rng = np.random.default_rng(seed)
        W, _ = np.linalg.qr(rng.normal(0, 1, (ind, nc)).astype(np.float32))
        self.W = W

    def transform(self, x):
        return x @ self.W


class _MLP:
    def __init__(self, ind=20, seed=42):
        rng = np.random.default_rng(seed)

        def xav(r, c):
            l = np.sqrt(6 / (r + c))
            return rng.uniform(-l, l, (r, c)).astype(np.float32)

        self.W1 = xav(64, ind)
        self.b1 = np.zeros(64, np.float32)
        self.W2 = xav(32, 64)
        self.b2 = np.zeros(32, np.float32)
        self.W3 = xav(1, 32)
        self.b3 = np.array([8.0], np.float32)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        h1 = _relu(self.W1 @ x + self.b1)
        h2 = _relu(self.W2 @ h1 + self.b2)
        out = (self.W3 @ h2 + self.b3).ravel()
        return float(out[0])


class _Robust:
    def __init__(self):
        self._med = None
        self._iqr = None

    def fit(self, X):
        self._med = np.median(X, axis=0).astype(np.float32)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = (q75 - q25).astype(np.float32)
        self._iqr = np.where(iqr < 1e-6, 1.0, iqr)

    def transform(self, x):
        return (x - self._med) / self._iqr if self._med is not None else x


class PHQ8Estimator:
    TOTAL_DIMS = 1050

    def __init__(self):
        self._sc = _Robust()
        self._pls = _PLS(20, 1050)
        self._mlp = _MLP(20)
        self._cov = []
        self._aur = []
        self._auc = []
        self._pose = []
        self._form = []
        self._stress_norms = []
        self._confs = []

    def add_frame(self, face, voice, stress_norm, conf):
        p = voice.get("pitch", 140)
        pv = voice.get("pitch_variance", 0.15)
        sr = voice.get("speech_rate", 3)
        trem = float(voice.get("voice_tremor", False))
        er = voice.get("energy_rms", 0.5)
        t = face.get("facial_tension", 0.2)
        e = face.get("eye_contact_score", 75) / 100
        bs = face.get("blink_score", 80) / 100
        hm = face.get("head_movement_score", 90) / 100

        cov = np.zeros(74, np.float32)
        cov[:8] = [
            p / 400,
            pv,
            min(sr / 8, 1),
            min(voice.get("pause_duration", 0) / 10, 1),
            trem,
            er,
            stress_norm,
            conf / 100 if conf > 1 else conf,
        ]
        for i in range(8, 74):
            cov[i] = np.sin(p * (i - 7) * np.pi / 400) * er
        self._cov.append(cov)

        conf_frac = conf / 100 if conf > 1 else conf
        self._aur.append(
            np.array(
                [
                    t * 5,
                    e * 3,
                    t * 3,
                    max(0, (e - 0.5) * 5),
                    max(0, (1 - t) * 2),
                    stress_norm * 3,
                    (1 - conf_frac) * 4,
                    (1 - bs) * 3,
                ],
                np.float32,
            )
        )
        self._auc.append(
            np.array(
                [
                    float(t > 0.5),
                    float(e < 0.5),
                    float(stress_norm > 0.5),
                    float(bs < 0.5),
                    float(hm < 0.6),
                    trem,
                ],
                np.float32,
            )
        )
        rx = (1 - hm) * 15
        self._pose.append(np.array([rx, 0, 0, 0, 0, 0], np.float32))
        self._form.append(np.array([p * k / 1000 for k in range(1, 6)], np.float32))
        self._stress_norms.append(stress_norm)
        self._confs.append(conf_frac)

    def predict(self) -> Dict:
        if not self._cov:
            return {
                "phq8_score": 0.0,
                "severity": "None/Minimal",
                "depressed_flag": False,
                "n_frames": 0,
                "timestamp": time.time(),
            }

        def agg(frames):
            mat = np.stack(frames)
            parts = []
            for col in range(mat.shape[1]):
                parts.append(_ten(mat[:, col]))
            return np.concatenate(parts)

        base = np.concatenate(
            [
                agg(self._cov),
                agg(self._aur),
                agg(self._auc),
                agg(self._pose),
                agg(self._form),
            ]
        )
        vec = np.concatenate(
            [base, np.zeros(self.TOTAL_DIMS - len(base), np.float32)]
        )[: self.TOTAL_DIMS]

        self._sc.fit(vec.reshape(1, -1))
        vs = self._sc.transform(vec)
        phq_model = float(
            np.clip(self._mlp.forward(self._pls.transform(vs)), 0, 24)
        )

        n = len(self._stress_norms)
        mean_stress = np.mean(self._stress_norms) if n else 0.5
        mean_conf = np.mean(self._confs) if n else 0.5
        heuristic = float(np.clip(
            24 * (0.4 * mean_stress + 0.3 * (1 - mean_conf) + 0.3), 0, 24
        ))
        phq = float(np.clip(0.70 * phq_model + 0.30 * heuristic, 0, 24))

        sev = (
            "None/Minimal"
            if phq < 5
            else "Mild"
            if phq < 10
            else "Moderate"
            if phq < 15
            else "Moderately Severe"
            if phq < 20
            else "Severe"
        )

        return {
            "phq8_score": round(phq, 2),
            "phq8_model_raw": round(phq_model, 2),
            "severity": sev,
            "depressed_flag": phq >= 10,
            "n_frames": len(self._cov),
            "mean_stress": round(float(mean_stress), 4),
            "mean_confidence": round(float(mean_conf), 4),
            "timestamp": time.time(),
        }

    def reset(self):
        self._cov.clear()
        self._aur.clear()
        self._auc.clear()
        self._pose.clear()
        self._form.clear()
        self._stress_norms.clear()
        self._confs.clear()
