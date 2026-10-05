import time

class SilenceIndex:
    """Compute Silence Stress Index (SSI) based on vocal metrics.

    SSI combines latency (response time), pause duration, filler word rate,
    and interruption frequency into a single score (0-100).
    """

    def __init__(self, weight_latency=0.25, weight_pause=0.25, weight_filler=0.25, weight_interrupt=0.25):
        self.weight_latency = weight_latency
        self.weight_pause = weight_pause
        self.weight_filler = weight_filler
        self.weight_interrupt = weight_interrupt
        self.history = []  # store recent scores for smoothing
        self.max_history = 10

    def _normalize(self, value, max_value):
        """Normalize a metric to 0-100 range, higher means more stress."""
        if max_value == 0:
            return 0
        return min(100, (value / max_value) * 100)

    def compute_score(self, latency, pause_duration, filler_rate, interruption_rate):
        """Calculate raw SSI score from individual components.

        Args:
            latency (float): Time (seconds) before the candidate starts speaking after a question.
            pause_duration (float): Average pause duration between utterances.
            filler_rate (float): Ratio of filler words to total words (0-1).
            interruption_rate (float): Number of interruptions per minute.
        """
        # Define reasonable max thresholds for normalization
        max_latency = 5.0          # seconds
        max_pause = 3.0           # seconds
        max_filler = 0.3          # 30% filler words
        max_interrupt = 3.0        # 3 interruptions per minute

        latency_score = self._normalize(latency, max_latency)
        pause_score = self._normalize(pause_duration, max_pause)
        filler_score = self._normalize(filler_rate, max_filler)
        interrupt_score = self._normalize(interruption_rate, max_interrupt)

        raw_score = (
            latency_score * self.weight_latency +
            pause_score * self.weight_pause +
            filler_score * self.weight_filler +
            interrupt_score * self.weight_interrupt
        )
        return raw_score

    def update(self, latency, pause_duration, filler_rate, interruption_rate):
        """Update internal history with a new SSI value and return smoothed score.
        """
        raw = self.compute_score(latency, pause_duration, filler_rate, interruption_rate)
        self.history.append(raw)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        smoothed = sum(self.history) / len(self.history)
        return {
            "ssi_raw": raw,
            "ssi_smoothed": smoothed,
            "timestamp": time.time()
        }
