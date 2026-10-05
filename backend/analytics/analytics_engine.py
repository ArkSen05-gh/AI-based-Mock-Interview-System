import time
from .silence_index import SilenceIndex
from .confidence_index import ConfidenceIndex
from .spike_detection import SpikeDetector

class AnalyticsEngine:
    """Engine that processes multimodal metrics and computes analytics.

    It calculates:
    - Silence Stress Index (SSI) via :class:`SilenceIndex`
    - Confidence Instability Index (CII) via :class:`ConfidenceIndex`
    - Spike alerts using :class:`SpikeDetector`
    The ``process`` method returns a dict aggregating these values.
    """

    def __init__(self):
        self.ssi_calculator = SilenceIndex()
        self.cii_calculator = ConfidenceIndex()
        self.ssi_spike_detector = SpikeDetector(window_size=10, threshold=60, min_consecutive=3)
        self.cii_spike_detector = SpikeDetector(window_size=10, threshold=50, min_consecutive=3)

    def process(self, *, face_metrics: dict, voice_metrics: dict) -> dict:
        """Compute analytics from the latest facial and vocal metrics.

        Parameters
        ----------
        face_metrics: dict
            Dictionary returned by :class:`FaceAnalyzer`
        voice_metrics: dict
            Dictionary returned by :class:`VoiceAnalyzer`
        Returns
        -------
        dict
            Contains raw and smoothed SSI/CII values and any spike alerts.
        """
        # Silence Stress Index
        pause = voice_metrics.get('pause_duration', 0.0)
        latency = voice_metrics.get('response_latency', 0.0)
        filler_rate = voice_metrics.get('filler_rate', 0.0)
        interruption_rate = voice_metrics.get('interruption_rate', 0.0)
        ssi = self.ssi_calculator.update(latency, pause, filler_rate, interruption_rate)

        # Confidence Instability Index (proxy confidence from speech_rate)
        speech_rate = voice_metrics.get('speech_rate', 0.0)
        confidence_proxy = min(100.0, speech_rate * 20)  # simple heuristic
        cii = self.cii_calculator.update(confidence_proxy)

        # Spike detection on raw scores
        ssi_spike = self.ssi_spike_detector.update(ssi['ssi_raw'])
        cii_spike = self.cii_spike_detector.update(cii['cii_raw'])

        alerts = []
        if ssi_spike:
            alerts.append(ssi_spike)
        if cii_spike:
            alerts.append(cii_spike)

        return {
            'timestamp': time.time(),
            'ssi_raw': ssi['ssi_raw'],
            'ssi_smoothed': ssi['ssi_smoothed'],
            'cii_raw': cii['cii_raw'],
            'cii_smoothed': cii['cii_smoothed'],
            'alerts': alerts,
        }
