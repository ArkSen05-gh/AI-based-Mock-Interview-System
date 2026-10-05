import time
import collections
from typing import Deque, Dict, Any

class SpikeDetector:
    """Detect spikes based on rolling average over a fixed window.

    Usage:
        detector = SpikeDetector(window_size=10, threshold=60)
        alert = detector.update(score)
    """
    def __init__(self, window_size: int = 10, threshold: float = 60.0, min_consecutive: int = 3):
        self.window_size = window_size
        self.threshold = threshold
        self.min_consecutive = min_consecutive
        self.scores: Deque[float] = collections.deque(maxlen=window_size)
        self.consecutive_hits = 0

    def update(self, score: float) -> Dict[str, Any] | None:
        """Add a new score and return an alert if conditions are met.

        Returns a dict with alert details or None.
        """
        self.scores.append(score)
        avg_score = sum(self.scores) / len(self.scores)
        if avg_score >= self.threshold:
            self.consecutive_hits += 1
            if self.consecutive_hits >= self.min_consecutive:
                self.consecutive_hits = 0
                return {
                    "type": "SPIKE_ALERT",
                    "score": avg_score,
                    "window": list(self.scores),
                    "timestamp": time.time(),
                    "message": f"Score spike detected (avg={avg_score:.1f})"
                }
        else:
            self.consecutive_hits = 0
        return None
