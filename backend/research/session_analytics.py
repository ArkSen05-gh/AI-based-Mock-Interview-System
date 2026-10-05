"""
Session Analytics — Mandal et al. (2024)
Eqs. 5.4 RMSE, 5.5 MAE, 5.6 CCC, 5.7 depression flag, 5.8 F1
Eqs. 5.10-5.12 population stress trajectory
Eq. 5.13 Pearson r, Eq. 5.14 spike z-score, Eq. 5.15 changepoints
"""
import numpy as np
import time
import logging
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger("session_analytics")


def rmse(yt, yp):
    return float(np.sqrt(np.mean((np.array(yt) - np.array(yp)) ** 2)))


def mae(yt, yp):
    return float(np.mean(np.abs(np.array(yt) - np.array(yp))))


def ccc(yt, yp):
    """Eq. 5.6 — population variance (N denominator)."""
    yt, yp = np.array(yt, float), np.array(yp, float)
    mu_t, mu_p = yt.mean(), yp.mean()
    var_t = np.var(yt)
    var_p = np.var(yp)
    cov = np.mean((yt - mu_t) * (yp - mu_p))
    d = var_t + var_p + (mu_t - mu_p) ** 2
    return round(float(2 * cov / d), 4) if d > 1e-9 else 0.0


def f1_depression(yt, yp, thresh=10.0):
    """Eq. 5.7+5.8"""
    yt, yp = np.array(yt) >= thresh, np.array(yp) >= thresh
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    pr = tp / (tp + fp) if tp + fp else 0.0
    re = tp / (tp + fn) if tp + fn else 0.0
    return round(2 * pr * re / (pr + re), 4) if pr + re else 0.0


def pearson_r(a, b):
    """Eq. 5.13"""
    a, b = np.array(a, float), np.array(b, float)
    ma, mb = a.mean(), b.mean()
    num = np.sum((a - ma) * (b - mb))
    den = np.sqrt(np.sum((a - ma) ** 2) * np.sum((b - mb) ** 2))
    return round(float(num / den), 4) if den > 1e-9 else 0.0


def resample_to_100(series):
    """Eq. 5.10"""
    if not series:
        return np.zeros(100, np.float32)
    arr = np.array(series, np.float32)
    return np.interp(
        np.linspace(0, 100, 100), np.linspace(0, 100, len(arr)), arr
    ).astype(np.float32)


def population_stress_stats(curves):
    """Eqs. 5.11-5.12"""
    if not curves:
        return {"mean": [], "std": [], "upper": [], "lower": [], "n": 0}
    M = np.stack([resample_to_100(c) for c in curves])
    mu = np.mean(M, axis=0)
    sig = np.sqrt(np.mean((M - mu) ** 2, axis=0))
    return {
        "mean": mu.tolist(),
        "std": sig.tolist(),
        "upper": np.clip(mu + sig, 0, 1).tolist(),
        "lower": np.clip(mu - sig, 0, 1).tolist(),
        "n": len(curves),
    }


def detect_changepoints(series, penalty=1.0):
    """Eq. 5.15 — greedy PELT approx."""
    if len(series) < 10:
        return []
    arr = np.array(series, np.float32)
    n = len(arr)
    w = max(5, n // 20)
    cps = []
    for i in range(w, n - w):
        l = arr[max(0, i - w) : i]
        r = arr[i : min(n, i + w)]
        base = np.var(arr[max(0, i - w) : min(n, i + w)])
        if (
            base - np.var(l) - np.var(r) > penalty * 0.05
            and (not cps or i - cps[-1] > w)
        ):
            cps.append(i)
    return cps


class SessionAnalytics:
    def __init__(self):
        self._stress = []
        self._conf_proxy = []
        self._conf_cnn = []
        self._fsm = []
        self._timestamps = []
        self._stress_spikes = []
        self._phq8 = None
        self._t0 = time.time()

    def record(
        self, stress_norm, conf_proxy, conf_cnn, fsm_state, timestamp=None
    ):
        self._stress.append(stress_norm)
        self._conf_proxy.append(conf_proxy)
        self._conf_cnn.append(conf_cnn)
        self._fsm.append(fsm_state)
        self._timestamps.append(timestamp or time.time())

    def record_spike(self, spike: dict):
        if spike is not None:
            self._stress_spikes.append(spike)

    def set_phq8(self, v):
        self._phq8 = v

    def finalize(self) -> Dict:
        n = len(self._stress)
        if not n:
            return {"error": "No data"}
        s = np.array(self._stress, np.float32)
        cc = np.array(self._conf_cnn, np.float32) if self._conf_cnn else np.zeros(n, np.float32)
        cp = np.array(self._conf_proxy, np.float32) if self._conf_proxy else np.zeros(n, np.float32)

        sc = Counter(self._fsm)
        pct = {k: round(v / n * 100, 2) for k, v in sc.items()}

        elapsed = time.time() - self._t0
        t0 = self._timestamps[0] if self._timestamps else 0
        series_seconds = [round(t - t0, 2) for t in self._timestamps]

        return {
            "duration_seconds": round(elapsed, 1),
            "n_frames": n,
            "stress_mean": round(float(s.mean()), 4),
            "stress_std": round(float(s.std()), 4),
            "stress_max": round(float(s.max()), 4),
            "confidence_mean": round(float(cc.mean()), 4),
            "confidence_min": round(float(cc.min()), 4),
            "pearson_r_au_cnn": pearson_r(cp, cc),
            "changepoints": detect_changepoints(self._stress),
            "stress_spikes": self._stress_spikes,
            "fsm_state_pct": pct,
            "fsm_state_series": self._fsm,
            "phq8_score": self._phq8,
            "stress_curve_100": resample_to_100(self._stress).tolist(),
            "confidence_curve_100": resample_to_100(self._conf_cnn).tolist()
            if self._conf_cnn
            else [],
            "stress_series": s.tolist(),
            "confidence_series": cc.tolist(),
            "fsm_state_series": self._fsm,
            "time_seconds": series_seconds,
            "timestamp": time.time(),
        }
