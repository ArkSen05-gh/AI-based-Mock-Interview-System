"""
AU-CNN Confidence Estimator — Mandal et al. (2024)
Eq. 3.3: Confidence(t) = 0.40*AU12r + 0.30*(1/(1+|Rx|)) + 0.20*(1-AU04r) + 0.10*AU45c
"""
import numpy as np
import time
import logging
from collections import deque
from typing import Dict, List

logger = logging.getLogger("au_cnn")


def _relu(x):
    return np.maximum(x, 0)


class _Conv1D:
    def __init__(self, ic, oc, k, seed):
        rng = np.random.default_rng(seed)
        lim = np.sqrt(2.0 / (ic * k))
        self.W = rng.normal(0, lim, (oc, ic, k)).astype(np.float32)
        self.b = np.zeros(oc, np.float32)
        self.k = k

    def forward(self, X):
        T, C = X.shape
        pad = np.zeros((self.k - 1, C), np.float32)
        Xp = np.concatenate([pad, X], axis=0)
        out = np.zeros((T, self.W.shape[0]), np.float32)
        for t in range(T):
            seg = Xp[t : t + self.k].T
            out[t] = np.einsum("ijk,jk->i", self.W, seg) + self.b
        return _relu(out)


class _NumpyCNN:
    IN = 9

    def __init__(self):
        self.c1 = _Conv1D(self.IN, 128, 3, 10)
        self.c2 = _Conv1D(128, 128, 5, 11)
        self.c3 = _Conv1D(128, 128, 7, 12)
        rng = np.random.default_rng(13)
        self.W = rng.normal(0, 0.05, (1, 128)).astype(np.float32)
        self.b = np.array([0.5], np.float32)

    def forward(self, X):
        x = self.c3.forward(self.c2.forward(self.c1.forward(X)))
        logit = (self.W @ x.mean(axis=0) + self.b).ravel()
        return float(1 / (1 + np.exp(-float(logit[0]))))


def au_confidence_proxy(au12r, rx, au04r, au45c):
    """Eq. 3.3"""
    au12r = np.clip(au12r / 5.0, 0, 1)
    au04r = np.clip(au04r / 5.0, 0, 1)
    return float(
        np.clip(
            0.40 * au12r + 0.30 / (1 + abs(rx)) + 0.20 * (1 - au04r) + 0.10 * float(bool(au45c)),
            0,
            1,
        )
    )


class AUConfidenceEstimator:
    CNN_WIN = 100

    def __init__(self):
        self.cnn = _NumpyCNN()
        self._buf = deque(maxlen=self.CNN_WIN)
        self.history = []

    def _to_row(self, face):
        t = face.get("facial_tension", 0.2)
        e = face.get("eye_contact_score", 75) / 100
        bs = face.get("blink_score", 80) / 100
        hm = face.get("head_movement_score", 90) / 100
        return np.array(
            [
                t * 5,
                e * 3,
                t * 3,
                max(0, (e - 0.5) * 5),
                float(bs < 0.6),
                (1 - hm) * 15,
                0,
                0,
                e,
            ],
            dtype=np.float32,
        )

    def update(self, face) -> Dict:
        row = self._to_row(face)
        self._buf.append(row)
        au12r, rx, au04r, au45c = row[3], row[5], row[0], row[4]
        proxy = au_confidence_proxy(au12r, rx, au04r, au45c)
        cnn_c = (
            self.cnn.forward(np.stack(list(self._buf)))
            if len(self._buf) >= 20
            else proxy
        )
        r = {
            "confidence_proxy": round(proxy, 4),
            "confidence_cnn": round(float(cnn_c), 4),
            "au04_r": round(float(au04r), 3),
            "au12_r": round(float(au12r), 3),
            "au45_c": int(au45c),
            "head_rx": round(float(rx), 2),
            "timestamp": time.time(),
        }
        self.history.append(r)
        return r

    def pearson_r(self):
        if len(self.history) < 5:
            return 0.0
        p = np.array([h["confidence_proxy"] for h in self.history])
        c = np.array([h["confidence_cnn"] for h in self.history])
        num = np.sum((p - p.mean()) * (c - c.mean()))
        den = np.sqrt(np.sum((p - p.mean()) ** 2) * np.sum((c - c.mean()) ** 2))
        return round(float(num / den), 4) if den > 1e-9 else 0.0
