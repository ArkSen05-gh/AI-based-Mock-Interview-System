"""
FSM Adaptive Questioning Controller — Mandal et al. (2024)
Eq. 3.4: pct_state = (seconds_in_state / total_seconds) * 100
Five states: Baseline | Monitor | Adapt | Recover | Escalate
"""
import time
import logging
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("fsm_controller")


class FSMState(Enum):
    BASELINE = "Baseline"
    MONITOR = "Monitor"
    ADAPT = "Adapt"
    RECOVER = "Recover"
    ESCALATE = "Escalate"


# Maximum total questions an interview may ask. Once reached, the interview
# MUST end regardless of the current FSM state — the FSM never loops
# indefinitely and no more than N_MAX_QUESTIONS are ever delivered.
N_MAX_QUESTIONS = 8


FSM_STATE_ID = {
    FSMState.BASELINE: 0,
    FSMState.MONITOR: 1,
    FSMState.ADAPT: 2,
    FSMState.RECOVER: 3,
    FSMState.ESCALATE: 4,
}

QUESTION_BANKS = {
    "general": {
        FSMState.BASELINE: [
            "Tell me about yourself and your professional background.",
            "What are your key strengths and how have they helped in your career?",
            "Describe a project you are particularly proud of.",
            "Where do you see yourself in five years?",
            "What motivates you most in your work?",
        ],
        FSMState.MONITOR: [
            "Can you elaborate on that experience in more detail?",
            "What was your specific role in that project?",
            "How did you handle competing priorities in that role?",
        ],
        FSMState.ADAPT: [
            "Take your time — describe one specific achievement from that role.",
            "Let's focus: what is the single biggest impact you've made in a team?",
            "In a few words, what makes you unique as a candidate?",
        ],
        FSMState.RECOVER: [
            "That's great context. How did it shape your approach to work?",
            "What would you say you learned most from that situation?",
        ],
        FSMState.ESCALATE: [
            "Let's try a different angle — what motivates you most professionally?",
        ],
    },
    "software": {
        FSMState.BASELINE: [
            "Explain the difference between a process and thread.",
            "How does garbage collection work in your primary language?",
            "Describe the CAP theorem and its trade-offs.",
            "Walk me through designing a REST API from scratch.",
            "Explain the SOLID principles with a concrete example.",
        ],
        FSMState.MONITOR: [
            "How would you handle race conditions in that scenario?",
            "What data structures would you choose and why?",
            "How would you optimise that query for large datasets?",
        ],
        FSMState.ADAPT: [
            "Describe a bug you found and how you debugged it.",
            "What is the simplest caching strategy you've implemented?",
            "Explain a design pattern you use regularly.",
        ],
        FSMState.RECOVER: [
            "Good. How would you test the solution you just described?",
            "What monitoring would you add to that system in production?",
        ],
        FSMState.ESCALATE: [
            "Describe your typical daily development workflow.",
        ],
    },
    "data_science": {
        FSMState.BASELINE: [
            "Explain the bias-variance tradeoff and how you manage it.",
            "How do you handle class imbalance in a classification problem?",
            "Walk me through your feature engineering process.",
            "Compare L1 and L2 regularisation — when would you use each?",
            "Describe a model you deployed to production and its impact.",
        ],
        FSMState.MONITOR: [
            "What cross-validation strategy for time-series data?",
            "How would you interpret a SHAP value for a tree-based model?",
            "Describe how you'd detect data drift in production.",
        ],
        FSMState.ADAPT: [
            "How do you evaluate model performance? Give an example.",
            "What is precision vs recall — give a real example.",
            "How would you explain model results to a non-technical stakeholder?",
        ],
        FSMState.RECOVER: [
            "How would you deploy that model and monitor it over time?",
            "What tools do you typically use for experiment tracking?",
        ],
        FSMState.ESCALATE: [
            "What is overfitting and how do you prevent it?",
        ],
    },
}


class FSMAdaptiveController:
    ADAPT_ESC_SECS = 30
    SILENCE_RECOVER = 3.0

    def __init__(self, domain="general"):
        self.domain = domain if domain in QUESTION_BANKS else "general"
        self.state = FSMState.BASELINE
        self._t = 0
        self._total = 0
        self._counts = {s: 0 for s in FSMState}
        self._q_idx = {d: {s: 0 for s in FSMState} for d in QUESTION_BANKS}
        self._log = []
        self._start = time.time()
        self._last_question = ""

    def _go(self, new, reason):
        if new == self.state:
            return
        self._log.append(
            {
                "from": self.state.value,
                "to": new.value,
                "reason": reason,
                "elapsed": round(time.time() - self._start, 1),
            }
        )
        logger.info(f"FSM {self.state.value}→{new.value} ({reason})")
        self.state = new
        self._t = 0

    def step(
        self, stress_norm: float, conf: float, silence: float = 0.0
    ) -> Dict:
        self._t += 1
        self._total += 1
        self._counts[self.state] += 1
        s = stress_norm
        c = conf / 100.0 if conf > 1.0 else conf

        if self.state == FSMState.BASELINE:
            if s > 0.65 or c < 0.35:
                self._go(FSMState.ADAPT, f"s={s:.2f} c={c:.2f}")
            elif s > 0.45:
                self._go(FSMState.MONITOR, f"s={s:.2f}")
        elif self.state == FSMState.MONITOR:
            if s > 0.65 or c < 0.35:
                self._go(FSMState.ADAPT, f"s={s:.2f} c={c:.2f}")
            elif s < 0.35:
                self._go(FSMState.BASELINE, f"recovered s={s:.2f}")
        elif self.state == FSMState.ADAPT:
            if self._t >= self.ADAPT_ESC_SECS:
                self._go(FSMState.ESCALATE, "30s in Adapt")
            elif s < 0.40 or silence > self.SILENCE_RECOVER:
                self._go(
                    FSMState.RECOVER, f"s={s:.2f} sil={silence:.1f}s"
                )
        elif self.state == FSMState.RECOVER:
            if s < 0.35:
                self._go(FSMState.BASELINE, f"fully recovered s={s:.2f}")
        elif self.state == FSMState.ESCALATE:
            self._go(FSMState.RECOVER, "auto-recover")

        q = self.current_question()
        pct = self._pct()

        return {
            "fsm_state": self.state.value,
            "fsm_state_id": FSM_STATE_ID[self.state],
            "t_in_state": self._t,
            "question": q,
            "pct_states": pct,
            "n_transitions": len(self._log),
            "timestamp": time.time(),
        }

    def current_question(self) -> str:
        """Return the question for the current FSM state WITHOUT advancing."""
        bank = QUESTION_BANKS[self.domain]
        ql = bank.get(self.state, bank[FSMState.BASELINE])
        idx = self._q_idx[self.domain][self.state] % len(ql)
        return ql[idx]

    def ask_question(self) -> str:
        """Advance the per-state question index and return the question.
        Called when the interviewer actually presents the next question."""
        bank = QUESTION_BANKS[self.domain]
        ql = bank.get(self.state, bank[FSMState.BASELINE])
        idx = self._q_idx[self.domain][self.state] % len(ql)
        q = ql[idx]
        self._q_idx[self.domain][self.state] += 1
        self._last_question = q
        return q

    def _pct(self):
        tot = max(1, self._total)
        return {s.value: round(self._counts[s] / tot * 100, 2) for s in FSMState}

    def session_summary(self) -> Dict:
        return {
            "total_seconds": self._total,
            "final_state": self.state.value,
            "final_state_id": FSM_STATE_ID[self.state],
            "pct_states": self._pct(),
            "transition_log": self._log,
            "n_transitions": len(self._log),
            "last_question": self._last_question,
        }
