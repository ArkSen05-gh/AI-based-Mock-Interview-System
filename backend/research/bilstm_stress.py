"""
BiLSTM Temporal Stress Model — Mandal et al. (2024)
Eq. 3.1: Stress(t) = 0.30*sigma(F0) + 0.20*|dF0| + 0.20*sigma(E) + 0.20*(1-VUV) + 0.10*(F1/1000)
Eq. 3.2: Stress_norm = (Stress - min) / (max - min)
Eq. 5.14: spike if z(t) > 2.0
"""
import numpy as np
import time
import logging
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger("bilstm_model")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


def _tanh(x):
    return np.tanh(np.clip(x, -15, 15))


class _LSTMCell:
    def __init__(self, ins, hid, seed):
        rng = np.random.default_rng(seed)

        def xav(r, c):
            lim = np.sqrt(6.0 / (r + c))
            return rng.uniform(-lim, lim, (r, c)).astype(np.float32)

        self.h = hid
        self.Wi = xav(hid, ins)
        self.Ui = xav(hid, hid)
        self.bi = np.zeros(hid, np.float32)
        self.Wf = xav(hid, ins)
        self.Uf = xav(hid, hid)
        self.bf = np.ones(hid, np.float32)
        self.Wg = xav(hid, ins)
        self.Ug = xav(hid, hid)
        self.bg = np.zeros(hid, np.float32)
        self.Wo = xav(hid, ins)
        self.Uo = xav(hid, hid)
        self.bo = np.zeros(hid, np.float32)

    def forward(self, X):
        h = np.zeros(self.h, np.float32)
        c = np.zeros(self.h, np.float32)
        outs = []
        for x in X:
            i = _sig(self.Wi @ x + self.Ui @ h + self.bi)
            f = _sig(self.Wf @ x + self.Uf @ h + self.bf)
            g = _tanh(self.Wg @ x + self.Ug @ h + self.bg)
            o = _sig(self.Wo @ x + self.Uo @ h + self.bo)
            c = f * c + i * g
            h = o * _tanh(c)
            outs.append(h.copy())
        return np.stack(outs)


class NumpyBiLSTM:
    def __init__(self, ins=24, hid=64):
        self.f1 = _LSTMCell(ins, hid, 0)
        self.b1 = _LSTMCell(ins, hid, 1)
        self.f2 = _LSTMCell(2 * hid, hid, 2)
        self.b2 = _LSTMCell(2 * hid, hid, 3)
        self.hid = hid

    def forward(self, X):
        l1 = np.concatenate(
            [self.f1.forward(X), self.b1.forward(X[::-1])[::-1]], axis=-1
        )
        return np.concatenate(
            [self.f2.forward(l1), self.b2.forward(l1[::-1])[::-1]], axis=-1
        )


class COVAREPStressProxy:
    """Eq. 3.1 + 3.2"""

    def __init__(self):
        self._mn = 1e9
        self._mx = -1e9

    def set_calibration_bounds(self, p10: float = None, p90: float = None):
        """FIX 4 — Cold-start: seed min/max normalisation bounds from the
        population calibration (stress_p10 / stress_p90) so the first stress
        value does not always normalise to 0."""
        if p10 is not None:
            self._mn = float(p10)
        if p90 is not None:
            self._mx = float(p90)

    def from_voice_metrics(self, v: dict) -> Dict[str, float]:
        pitch = v.get("pitch", 140.0)
        pv = v.get("pitch_variance", 0.15)
        sr = v.get("speech_rate", 3.0)
        trem = float(v.get("voice_tremor", False))
        erms = v.get("energy_rms", 0.5)

        sigma_f0 = pitch * pv
        delta_f0 = pitch * pv * 0.5
        sigma_e = erms * pv
        vuv = min(1.0, sr / 5.0)
        f1 = pitch * 1.5

        raw = float(
            np.clip(
                0.30 * sigma_f0 / 100
                + 0.20 * delta_f0 / 50
                + 0.20 * sigma_e
                + 0.20 * (1 - vuv)
                + 0.10 * (f1 / 1000)
                + 0.10 * trem,
                0,
                1,
            )
        )
        self._mn = min(self._mn, raw)
        self._mx = max(self._mx, raw)
        denom = self._mx - self._mn
        norm = float((raw - self._mn) / denom) if denom > 1e-6 else raw
        return {"stress_raw": round(raw, 4), "stress_norm": round(norm, 4)}


class StressSpikeDetector:
    """Eq. 5.14: z(t)=(stress-mu30)/sigma30, spike if z>2"""

    def __init__(self):
        self._buf = deque(maxlen=30)

    def update(self, sn: float) -> Optional[Dict]:
        self._buf.append(sn)
        if len(self._buf) < 5:
            return None
        arr = np.array(self._buf)
        mu = arr.mean()
        sig = arr.std()
        z = (sn - mu) / sig if sig > 1e-6 else 0.0
        if z > 2.0:
            return {
                "type": "STRESS_SPIKE",
                "z_score": round(z, 3),
                "stress_norm": round(sn, 4),
                "timestamp": time.time(),
                "message": f"Stress spike z={z:.2f}",
            }
        return None


class BiLSTMStressModel:
    FEATURE_DIM = 24
    HID = 64
    WINDOW = 30

    def __init__(self):
        self.proxy = COVAREPStressProxy()
        self.spike = StressSpikeDetector()
        self.bilstm = NumpyBiLSTM(self.FEATURE_DIM, self.HID)
        rng = np.random.default_rng(99)
        self.W_out = rng.normal(0, 0.05, (2, 2 * self.HID)).astype(np.float32)
        self.b_out = np.array([-1.0, 50.0], np.float32)
        self._win = deque(maxlen=self.WINDOW)
        self.history = []

    def _fv(self, face, voice, body, ctx, sn):
        p = body.get("posture", "open")
        v = np.array(
            [
                face.get("eye_contact_score", 75) / 100,
                face.get("blink_score", 80) / 100,
                face.get("head_movement_score", 90) / 100,
                face.get("facial_tension", 0.2),
                face.get("AU04", face.get("facial_tension", 0.2)),
                face.get(
                    "AU12",
                    max(0.0, (face.get("eye_contact_score", 75) - 50) / 50),
                ),
                voice.get("pitch", 140) / 400,
                voice.get("pitch_variance", 0.15),
                min(voice.get("speech_rate", 3) / 8, 1),
                min(voice.get("pause_duration", 0) / 10, 1),
                float(voice.get("voice_tremor", False)),
                voice.get("audio_stress_score", 20) / 100,
                voice.get("filler_rate", 0.05),
                voice.get("energy_rms", 0.5),
                float(body.get("fidgeting", False)),
                float(p == "open"),
                float(p == "closed"),
                float(p == "slouched"),
                ctx.get("ssi_raw", 20) / 100,
                ctx.get("cii_raw", 10) / 100,
                min(ctx.get("response_latency", 1) / 5, 1),
                ctx.get("silence_ratio", 0.2),
                sn,
                float(ctx.get("fsm_state_id", 0)) / 4,
            ],
            dtype=np.float32,
        )
        return np.clip(v, -3, 3)

    def predict(self, face, voice, body, ctx) -> Dict:
        stress = self.proxy.from_voice_metrics(voice)
        sn = stress["stress_norm"]
        spike_alert = self.spike.update(sn)

        fv = self._fv(face, voice, body, ctx, sn)
        self._win.append(fv)

        if len(self._win) >= 5:
            hidden = self.bilstm.forward(np.stack(list(self._win)))
            logit = self.W_out @ hidden[-1] + self.b_out
            sb = float(_sig(logit[0]))
            cb = float(np.clip(50 + _tanh(logit[1] / 50) * 50, 0, 100))
        else:
            sb = sn
            cb = face.get("eye_contact_score", 75)

        r = {
            "stress_proxy_raw": stress["stress_raw"],
            "stress_proxy_norm": sn,
            "stress_bilstm": round(sb, 4),
            "confidence_bilstm": round(cb, 2),
            "spike_alert": spike_alert,
            "timestamp": time.time(),
        }
        self.history.append(r)
        return r

    def get_history(self):
        return list(self.history)

    def get_stress_series(self):
        return [h["stress_proxy_norm"] for h in self.history]

    def get_confidence_series(self):
        return [h["confidence_bilstm"] for h in self.history]
