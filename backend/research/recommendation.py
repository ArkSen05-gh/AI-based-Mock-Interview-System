"""
Recommendation Report — LLM-generated interview recommendation (Section 6).

Calls Groq (OpenAI-compatible endpoint) with the full session metrics and
returns the structured recommendation report. Degrades gracefully to a
deterministic, rule-based report when a Groq API key is missing or the call
fails, so the endpoint never 500s.
"""
import os
import json
import logging
from typing import Dict, Optional, List

logger = logging.getLogger("recommendation")

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1500

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_performance": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["Strong", "Moderate", "Concerning", "High-Risk"]},
                "summary": {"type": "string"},
            },
            "required": ["verdict", "summary"],
        },
        "stress_profile": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "recommendations"],
        },
        "confidence_profile": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "recommendations"],
        },
        "adapting_recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "disclaimer": {"type": "string"},
    },
    "required": [
        "overall_performance", "stress_profile", "confidence_profile",
        "adapting_recommendations", "next_steps", "disclaimer",
    ],
}


def _build_prompt(summary: Dict, metrics: Optional[Dict] = None) -> str:
    stress_mean = summary.get("stress_mean", 0)
    conf_mean = summary.get("confidence_mean", 0)
    pearson = summary.get("pearson_r_au_cnn", 0)
    phq8 = (summary.get("phq8") or {}).get("phq8_score", summary.get("phq8_score", 0))
    n_transitions = (summary.get("fsm_session") or {}).get("n_transitions", 0)
    final_state = (summary.get("fsm_session") or {}).get("final_state", "Unknown")
    st_spikes = len([r for r in summary.get("per_second", []) if r.get("spike_alert")])
    pct = (summary.get("fsm_state_pct") or {})
    domain = summary.get("domain", "general")

    return (
        "You are an expert behavioral-interview analyst for a hiring decision support system.\n"
        "Given the multimodal metrics from a candidate's live interview, produce a structured "
        "recommendation report as JSON strictly matching the provided schema.\n\n"
        "IMPORTANT: stress confidence and PHQ-8 signals are COMPUTATIONAL PROXIES derived from "
        "voice/face cues for research and evaluation only. They are NOT clinical diagnoses.\n\n"
        f"Domain: {domain}\n"
        f"Mean stress (0-1): {stress_mean:.3f}\n"
        f"Mean confidence (0-1): {conf_mean:.3f}\n"
        f"AU-CNN proxy-vs-CNN Pearson r: {pearson:.3f}\n"
        f"PHQ-8 proxy score (0-24, research only): {phq8:.1f}\n"
        f"FSM final state: {final_state}\n"
        f"FSM transitions: {n_transitions}\n"
        f"Stress spikes: {st_spikes}\n"
        f"FSM state distribution: {json.dumps(pct)}\n"
        "Write concise, actionable, human-resources-oriented recommendations. "
        "Return ONLY a single JSON object matching the schema."
    )


def _rule_fallback(summary: Dict) -> Dict:
    stress_mean = summary.get("stress_mean", 0)
    conf_mean = summary.get("confidence_mean", 0)
    phq8 = (summary.get("phq8") or {}).get("phq8_score", summary.get("phq8_score", 0))
    n_trans = (summary.get("fsm_session") or {}).get("n_transitions", 0)

    if stress_mean > 0.65 or conf_mean < 0.35:
        verdict = "High-Risk"
        overall = "Elevated stress / low confidence proxies require a careful, structured follow-up."
    elif stress_mean > 0.45 or conf_mean < 0.5:
        verdict = "Concerning"
        overall = "Moderate stress patterns observed; probe resilience with targeted questions."
    elif stress_mean > 0.35:
        verdict = "Moderate"
        overall = "Balanced performance with some stress signals under adaptive questioning."
    else:
        verdict = "Strong"
        overall = "Candidate demonstrated calm, confident delivery throughout the interview."

    return {
        "overall_performance": {
            "verdict": verdict,
            "summary": overall,
        },
        "stress_profile": {
            "summary": f"Mean computational stress proxy {stress_mean:.2f} with {n_trans} FSM state transitions.",
            "recommendations": [
                "Ask behavioral questions about pressure handling and recovery from setbacks.",
                "Note stress spikes and confirm whether they reflect domain difficulty vs. dispositional anxiety.",
            ],
        },
        "confidence_profile": {
            "summary": f"Mean computational confidence proxy {conf_mean:.2f}.",
            "recommendations": [
                "Provide clear structure and positive reinforcement to sustain engagement.",
                "Validate technical depth with open-ended follow-ups where confidence dipped.",
            ],
        },
        "adapting_recommendations": [
            "Use a Monitor/Adapt state question bank tailored to observed stress level.",
            "Keep question difficulty calibrated to avoid sustained escalation (Adapt = 30s rule).",
        ],
        "next_steps": [
            "Conduct a structured behavioral panel with one resilience-focused scenario.",
            "Triangulate the automated recommendation with a human interviewer debrief.",
        ],
        "disclaimer": "This report is generated from computational proxies for research/evaluation "
                      "support and is NOT a clinical or hiring decision on its own.",
    }


def generate_recommendation(
    summary: Dict,
    metrics: Optional[Dict] = None,
    api_key: Optional[str] = None,
) -> Dict:
    """Generate the structured recommendation report via Groq (OpenAI-compatible).
    Falls back to the deterministic rule report on any failure."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        logger.warning("No GROQ_API_KEY set; using rule-based recommendation fallback.")
        return _rule_fallback(summary)

    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce structured JSON-only recommendations for interview decision "
                        "support. Never rely on computational proxies as clinical facts. "
                        "Emit exactly one JSON object."
                    ),
                },
                {"role": "user", "content": _build_prompt(summary, metrics)},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        report = json.loads(text)
        if not isinstance(report, dict):
            raise ValueError("Groq returned a non-object response")
        logger.info(f"Groq recommendation generated with {GROQ_MODEL}")
        return report
    except Exception as e:
        logger.error(f"Groq recommendation failed ({e}); using rule fallback.")
        return _rule_fallback(summary)
