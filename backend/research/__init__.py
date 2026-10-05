"""
Research Module — BiLSTM-Based Temporal Stress Forecasting with FSM-Driven
Adaptive Questioning (Mandal et al., 2024).

Packages the paper's models so they can run both:
  * real-time  (inside the FastAPI websocket, via :class:`InterviewEngine`)
  * offline    (batch analysis over recorded COVAREP / OpenFace / transcript files,
                see ``offline_analysis.py`` and ``data_adapter.py``)

Public API:
    InterviewEngine      - per-candidate orchestrator (stress, confidence, FSM, PHQ-8)
    BiLSTMStressModel    - Eq. 3.1 / 3.2 stress proxy + temporal BiLSTM
    AUConfidenceEstimator- Eq. 3.3 AU-CNN confidence estimator
    FSMAdaptiveController- Eq. 3.4 five-state adaptive questioning controller
    PHQ8Estimator        - Eqs. 4.1-4.3, 5.1-5.3 session-level PHQ-8 regression
    SessionAnalytics     - Eqs. 5.4-5.15 evaluation metrics & population curves
"""
from .bilstm_stress import BiLSTMStressModel
from .au_cnn_confidence import AUConfidenceEstimator
from .fsm_controller import FSMAdaptiveController, FSMState
from .phq8_estimator import PHQ8Estimator
from .session_analytics import (
    rmse,
    mae,
    ccc,
    f1_depression,
    pearson_r,
    resample_to_100,
    population_stress_stats,
    detect_changepoints,
    SessionAnalytics,
)
from .interview_engine import InterviewEngine

__all__ = [
    "InterviewEngine",
    "BiLSTMStressModel",
    "AUConfidenceEstimator",
    "FSMAdaptiveController",
    "FSMState",
    "PHQ8Estimator",
    "SessionAnalytics",
    "rmse",
    "mae",
    "ccc",
    "f1_depression",
    "pearson_r",
    "resample_to_100",
    "population_stress_stats",
    "detect_changepoints",
]