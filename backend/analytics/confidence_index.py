import time
from collections import deque
from .metrics import InterviewMetrics

class ConfidenceIndex:
    """Compute Confidence Instability Index (CII).

    CII measures fluctuations in confidence scores over recent answers.
    Uses a rolling window and returns a normalized score (0‑100).
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.confidences = deque(maxlen=window_size)
        self.history = []  # recent CII values for optional smoothing

    def update(self, confidence: float) -> dict:
        """Add a new confidence value and compute CII.
        Returns raw, smoothed scores and timestamp.
        """
        confidence = max(0.0, min(100.0, confidence))
        self.confidences.append(confidence)
        if len(self.confidences) < 2:
            raw = 0.0
        else:
            mean = sum(self.confidences) / len(self.confidences)
            variance = sum((c - mean) ** 2 for c in self.confidences) / len(self.confidences)
            std_dev = variance ** 0.5
            raw = min(100.0, (std_dev / 50.0) * 100.0)  # assuming max std_dev 50
        self.history.append(raw)
        smoothed = sum(self.history[-5:]) / min(len(self.history), 5)
        return {
            "cii_raw": round(raw, 2),
            "cii_smoothed": round(smoothed, 2),
            "timestamp": time.time()
        }
