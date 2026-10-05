"""
Interview Engine — Per-candidate orchestrator for real-time research analysis.

Integrates BiLSTM Stress, AU-CNN Confidence, FSM Adaptive Controller,
PHQ-8 Estimator, and Session Analytics into a single step() interface
called from the FastAPI WebSocket handler.
"""
import time
import uuid
import logging
import numpy as np
from collections import deque
from typing import Dict, Optional

from .bilstm_stress import BiLSTMStressModel
from .au_cnn_confidence import AUConfidenceEstimator
from .fsm_controller import FSMAdaptiveController, FSM_STATE_ID
from .phq8_estimator import PHQ8Estimator
from .session_analytics import SessionAnalytics
from .feature_extractor import compute_filler_rate

logger = logging.getLogger("interview_engine")


class InterviewEngine:
    STEP_INTERVAL = 1.0  # min seconds between research steps (1 Hz)

    def __init__(self, domain: str = "general", session_id: str = None,
                 calibration: dict = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.domain = domain
        self._t0 = time.time()
        self._last_step_time = 0.0
        self._step_count = 0
        self._finalized = False
        self._summary_cache = None

        # Research components
        self.stress_model = BiLSTMStressModel()
        self.conf_estimator = AUConfidenceEstimator()
        self.fsm = FSMAdaptiveController(domain)
        self.phq8 = PHQ8Estimator()
        self.analytics = SessionAnalytics()

        # Apply calibration if provided
        if calibration:
            self._apply_calibration(calibration)

        # State tracking
        self._silence_acc = 0.0
        self._latency = 0.0
        self._speech_active = False
        self._filler_rate = 0.0
        self._silence_ratio = 0.2
        self._recent_speech = deque(maxlen=20)
        self._current_transcript = ""
        self._question_start_time = time.time()
        self._transcript_buffer = ""
        self._last_voice = {}
        self._last_face = {}

    def _apply_calibration(self, cal: dict):
        """Apply pre-trained calibration parameters to models."""
        bilstm_cal = cal.get("bilstm_calibration", cal)
        if "W_out" in bilstm_cal:
            self.stress_model.W_out = np.array(bilstm_cal["W_out"], np.float32)
            logger.info("Applied BiLSTM W_out calibration")
        if "b_out" in bilstm_cal:
            self.stress_model.b_out = np.array(bilstm_cal["b_out"], np.float32)

        # FIX 4 — cold-start the stress proxy normalisation bounds from the
        # population percentile calibration so the first stress value is
        # not always normalised to 0.
        p10 = bilstm_cal.get("stress_p10")
        p90 = bilstm_cal.get("stress_p90")
        if p10 is not None and p90 is not None:
            self.stress_model.proxy.set_calibration_bounds(p10, p90)
            logger.info(f"Applied stress proxy cold-start bounds (p10={p10}, p90={p90})")

        au_cnn_cal = cal.get("au_cnn_calibration", {})
        if au_cnn_cal:
            logger.info(f"Applied AU-CNN calibration (proxy_mean={au_cnn_cal.get('proxy_mean', 'N/A')})")

        phq8_cal = cal.get("phq8_calibration", {})
        if phq8_cal:
            logger.info(f"Applied PHQ-8 calibration (score={phq8_cal.get('phq8_score', 'N/A')})")

    def mark_question_start(self):
        """Call when a new question is presented."""
        self._question_start_time = time.time()
        self._latency = 0.0
        self._speech_active = False
        self._current_transcript = ""
        self._transcript_buffer = ""

    def update_transcript(self, text: str):
        """Feed partial or final transcript for filler rate / WPS."""
        self._transcript_buffer = text
        self._current_transcript = text
        self._filler_rate = compute_filler_rate(text)

    def should_step(self) -> bool:
        now = time.time()
        return (now - self._last_step_time) >= self.STEP_INTERVAL

    def step(
        self,
        face: dict,
        voice: dict,
        body: dict = None,
        ctx: dict = None,
    ) -> Optional[Dict]:
        """Run one research analysis step. Returns enriched metrics or None if throttled."""
        if self._finalized:
            return self._summary_cache

        if body is None:
            body = {}
        if ctx is None:
            ctx = {}

        now = time.time()
        self._last_voice = voice
        self._last_face = face

        # Silence / latency tracking
        energy = voice.get("energy_rms", 0.5)
        is_speech = energy > 0.05 and voice.get("speech_rate", 0) > 0.1
        self._recent_speech.append(is_speech)

        if is_speech:
            if not self._speech_active:
                self._latency = now - self._question_start_time
            self._speech_active = True
            self._silence_acc = 0.0
        else:
            self._speech_active = False
            self._silence_acc += self.STEP_INTERVAL

        if len(self._recent_speech) > 0:
            self._silence_ratio = sum(
                1 for s in self._recent_speech if not s
            ) / len(self._recent_speech)
        else:
            self._silence_ratio = 0.5

        # Merge context
        ctx["response_latency"] = self._latency
        ctx["silence_ratio"] = self._silence_ratio
        ctx["filler_rate"] = self._filler_rate
        ctx["fsm_state_id"] = FSM_STATE_ID.get(self.fsm.state, 0)

        # BiLSTM stress
        bilstm_result = self.stress_model.predict(face, voice, body, ctx)
        stress_norm = bilstm_result["stress_proxy_norm"]

        # AU-CNN confidence
        conf_result = self.conf_estimator.update(face)
        conf_cnn = conf_result["confidence_cnn"]
        conf_pct = conf_cnn * 100.0

        # FSM adaptive controller
        fsm_result = self.fsm.step(stress_norm, conf_pct, self._silence_acc)

        # PHQ-8 accumulation
        self.phq8.add_frame(face, voice, stress_norm, conf_cnn)

        # Session analytics
        conf_proxy = conf_result["confidence_proxy"]
        self.analytics.record(
            stress_norm, conf_proxy, conf_cnn,
            fsm_result["fsm_state"], now,
        )
        if bilstm_result.get("spike_alert"):
            self.analytics.record_spike(bilstm_result["spike_alert"])

        self._step_count += 1
        self._last_step_time = now

        return {
            "session_id": self.session_id,
            "step_index": self._step_count,
            "elapsed": round(now - self._t0, 2),
            "stress_proxy_raw": bilstm_result["stress_proxy_raw"],
            "stress_proxy_norm": stress_norm,
            "stress_pct": round(stress_norm * 100, 2),
            "stress_bilstm": bilstm_result["stress_bilstm"],
            "confidence_proxy": conf_result["confidence_proxy"],
            "confidence_cnn": conf_cnn,
            "confidence_pct": round(conf_pct, 2),
            "au_confidence": {
                "au04_r": conf_result["au04_r"],
                "au12_r": conf_result["au12_r"],
                "au45_c": conf_result["au45_c"],
                "head_rx": conf_result["head_rx"],
            },
            "fsm_state": fsm_result["fsm_state"],
            "fsm_state_id": fsm_result["fsm_state_id"],
            "fsm_question": fsm_result["question"],
            "fsm_t_in_state": fsm_result["t_in_state"],
            "fsm_pct_states": fsm_result["pct_states"],
            "spike_alert": bilstm_result.get("spike_alert"),
            "silence_ratio": round(self._silence_ratio, 3),
            "latency": round(self._latency, 3),
            "filler_rate": round(self._filler_rate, 4),
            "timestamp": now,
        }

    def finalize(self) -> Dict:
        """Compute final session summary. Idempotent."""
        if self._finalized and self._summary_cache is not None:
            return self._summary_cache

        # PHQ-8 prediction
        phq8_result = self.phq8.predict()
        self.analytics.set_phq8(phq8_result["phq8_score"])

        # Session analytics
        summary = self.analytics.finalize()
        summary["session_id"] = self.session_id
        summary["domain"] = self.domain
        summary["phq8"] = phq8_result
        summary["fsm_session"] = self.fsm.session_summary()
        summary["pearson_r_au_cnn"] = self.conf_estimator.pearson_r()

        # Stress spikes from BiLSTM
        bilstm_spikes = [
            h["spike_alert"]
            for h in self.stress_model.get_history()
            if h.get("spike_alert")
        ]
        summary["bilstm_spikes"] = bilstm_spikes

        self._finalized = True
        self._summary_cache = summary
        return summary
