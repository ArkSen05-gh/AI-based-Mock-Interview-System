from pydantic import BaseModel

class InterviewMetrics(BaseModel):
    """Pydantic model capturing per‑question interview metrics.

    Fields correspond to the data points collected during the interview and are
    used for SSI and CII calculations.
    """
    timestamp: float
    confidence: float
    pause_duration: float
    response_latency: float
    hesitation_count: int
    interrupted_speech: bool
    question_number: int
    filler_count: int = 0  # optional, used for SSI
