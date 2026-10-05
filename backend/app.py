# backend/app.py

import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import logging
import base64
from typing import Optional, List, Dict
from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
from analyzers.face_analyzer import FaceAnalyzer
from analyzers.voice_analyzer import VoiceAnalyzer
from analyzers.fusion_engine import MultimodalFusionEngine
from analytics.analytics_engine import AnalyticsEngine
from research.interview_engine import InterviewEngine
from research.fsm_controller import N_MAX_QUESTIONS
from google import genai
import asyncio
import interview_api

# ---------------- LOGGING ---------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview_backend")

# ---------------- APP ---------------- #

app = FastAPI(title="Multimodal Interview Evaluation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- TRAINED MODEL CALIBRATION ---------------- #

CALIBRATION_DATA: Optional[dict] = None
TRAINING_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")

def _load_calibration():
    """Load calibration data from trained_models/ if available."""
    global CALIBRATION_DATA
    report_path = os.path.join(TRAINING_OUTPUT_DIR, "calibration_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                CALIBRATION_DATA = json.load(f)
            logger.info(f"Loaded calibration data from {report_path}")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")
    else:
        logger.info("No calibration data found. Models use default weights.")

_load_calibration()

# ---------------- INTERVIEW REST API MANAGER ---------------- #

INTERVIEW_MGR = interview_api.build_manager(CALIBRATION_DATA)

# ---------------- MODELS ---------------- #

class AnswerItem(BaseModel):
    question: str
    answer: Optional[str] = ""

class NextQuestionRequest(BaseModel):
    domain: str
    previous_answer: str
    history: List[dict] = []
    session_id: Optional[str] = None
    question_count: int = 0

# ---------------- CONSTANTS ---------------- #

FILLER_WORDS = {
    "um", "uh", "like", "you", "know", "so",
    "actually", "basically", "erm", "ah"
}

GAZE_ALERT_THRESHOLD = 5.0      # seconds
STRESS_SPIKE_THRESHOLD = 60     # stress score

# ---------------- RESEARCH SESSION STORAGE ---------------- #

RESEARCH_ENGINES: Dict[str, InterviewEngine] = {}
RESEARCH_RESULTS: Dict[str, dict] = {}

# ---------------- HELPERS ---------------- #

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))

# ---------------- REAL-TIME DETECTION ---------------- #

def detect_stress_spike(metrics: dict):
    stress_score = 0
    wps = metrics.get("words_per_second", 0)
    filler_rate = metrics.get("filler_rate", 0)
    pause_duration = metrics.get("pause_duration", 0)

    if wps > 3.5:
        stress_score += 30
    if filler_rate > 0.20:
        stress_score += 30
    if pause_duration > 1.5:
        stress_score += 20

    if stress_score >= STRESS_SPIKE_THRESHOLD:
        return {
            "type": "STRESS_SPIKE",
            "level": stress_score,
            "timestamp": time.time(),
            "message": "Sudden stress spike detected"
        }
    return None


def detect_gaze_alert(gaze_data: dict):
    direction = gaze_data.get("gaze")
    duration = gaze_data.get("duration", 0)
    if direction in {"left", "right", "up", "down"} and duration >= GAZE_ALERT_THRESHOLD:
        return {
            "type": "GAZE_ALERT",
            "direction": direction,
            "duration": duration,
            "timestamp": time.time(),
            "message": "Look at the web Camera"
        }
    return None


def detect_nervousness(multimodal_data: dict):
    nervousness_score = 0
    facial = multimodal_data.get("facial_expressions", {})
    if facial.get("tension", 0) > 0.7 or facial.get("micro_expressions", "") == "fear":
        nervousness_score += 35
    vocal = multimodal_data.get("vocal_cues", {})
    if vocal.get("pitch_variance", 0) > 0.5 or vocal.get("tremor", False):
        nervousness_score += 35
    if nervousness_score >= 60:
        return {
            "type": "NERVOUSNESS_ALERT",
            "score": nervousness_score,
            "timestamp": time.time(),
            "message": "High nervousness detected"
        }
    return None

def process_realtime_metrics(data: dict):
    events = []
    if "facial_expressions" in data and "vocal_cues" in data:
        nerv = detect_nervousness(data)
        if nerv:
            events.append(nerv)
    if "words_per_second" in data:
        stress = detect_stress_spike(data)
        if stress:
            events.append(stress)
    if "gaze" in data:
        gaze = detect_gaze_alert(data)
        if gaze:
            events.append(gaze)
    return events

# ---------------- WEBSOCKET ---------------- #

active_connections: List[WebSocket] = []

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WebSocket client connected")

    # Initialize per-session analyzers
    face_analyzer = FaceAnalyzer()
    voice_analyzer = VoiceAnalyzer()
    fusion_engine = MultimodalFusionEngine()
    analytics_engine = AnalyticsEngine()

    # Research engine (created on init message)
    research_engine: Optional[InterviewEngine] = None
    session_id: Optional[str] = None

    last_face_metrics = {"eye_contact_score": 100.0, "blink_score": 100.0, "head_movement_score": 100.0, "facial_tension": 0.15}
    last_voice_metrics = {"speech_rate": 0.0, "pitch": 120.0, "pitch_variance": 0.1, "voice_tremor": False, "pause_duration": 0.0, "audio_stress_score": 0.0}
    latest_transcript = ""

    consecutive_gaze_deviations = 0
    consecutive_high_nervousness = 0

    try:
        while True:
            data = await websocket.receive_json()
            payload_type = data.get("type")

            # Handle init message (new research session)
            if payload_type == "init":
                session_id = data.get("session_id", str(time.time()))
                domain = data.get("domain", "general")
                research_engine = InterviewEngine(
                    domain=domain, session_id=session_id,
                    calibration=CALIBRATION_DATA,
                )
                RESEARCH_ENGINES[session_id] = research_engine
                logger.info(f"Research session started: {session_id} (domain={domain}, calibrated={CALIBRATION_DATA is not None})")
                await websocket.send_json({"type": "INIT_OK", "session_id": session_id, "calibrated": CALIBRATION_DATA is not None})
                continue

            # Handle transcript updates
            if payload_type == "text":
                text = data.get("text", "")
                latest_transcript = text
                if research_engine:
                    research_engine.update_transcript(text)
                continue

            # Handle question start
            if payload_type == "question_start":
                if research_engine:
                    research_engine.mark_question_start()
                continue

            # Handle video frames (base64 image)
            if payload_type == "video" or "image" in data:
                img_data = data.get("image")
                if img_data:
                    if "," in img_data:
                        img_data = img_data.split(",")[1]
                    try:
                        frame_bytes = base64.b64decode(img_data)
                        face_metrics = face_analyzer.process_frame(frame_bytes)
                        last_face_metrics = face_metrics

                        if face_metrics["eye_contact_score"] < 70.0:
                            consecutive_gaze_deviations += 1
                            if consecutive_gaze_deviations >= 15:
                                await websocket.send_json({
                                    "type": "ALERT",
                                    "alert_type": "GAZE_ALERT",
                                    "score": round(face_metrics["eye_contact_score"], 2),
                                    "message": "Please maintain eye contact with the camera."
                                })
                                consecutive_gaze_deviations = 0
                        else:
                            consecutive_gaze_deviations = max(0, consecutive_gaze_deviations - 1)
                    except Exception as e:
                        logger.error(f"Error processing video frame in WS: {e}")

            # Handle audio chunks (base64 sound blob)
            elif payload_type == "audio" or "audio" in data:
                audio_data = data.get("audio")
                if audio_data:
                    if "," in audio_data:
                        audio_data = audio_data.split(",")[1]
                    try:
                        audio_bytes = base64.b64decode(audio_data)
                        voice_metrics = voice_analyzer.process_audio(audio_bytes)
                        last_voice_metrics = voice_metrics

                        if voice_metrics["audio_stress_score"] >= 70.0:
                            await websocket.send_json({
                                "type": "ALERT",
                                "alert_type": "STRESS_SPIKE",
                                "score": round(voice_metrics["audio_stress_score"], 2),
                                "message": "Sudden vocal stress spike detected."
                            })
                    except Exception as e:
                        logger.error(f"Error processing audio chunk in WS: {e}")

            # Handle browser-side audio features (Web Audio API)
            elif payload_type == "audio_features":
                last_voice_metrics = {
                    "speech_rate": data.get("speech_rate", 0.0),
                    "pitch": data.get("pitch", 120.0),
                    "pitch_variance": data.get("pitch_variance", 0.1),
                    "voice_tremor": data.get("voice_tremor", 0.0) > 0.3,
                    "pause_duration": 0.0,
                    "energy_rms": data.get("energy_rms", 0.0),
                    "audio_stress_score": min(100.0, max(0.0, data.get("energy_rms", 0.0) * 800 + (data.get("pitch_variance", 0.0) * 40))),
                }
                continue  # Skip fusion/notify for audio-only updates

            # Handle end message (client closing session)
            elif payload_type == "end":
                if research_engine and session_id:
                    try:
                        summary = research_engine.finalize()
                        RESEARCH_RESULTS[session_id] = summary
                        if session_id in RESEARCH_ENGINES:
                            del RESEARCH_ENGINES[session_id]
                        logger.info(f"Research session finalized via 'end' message: {session_id}")
                    except Exception as e:
                        logger.error(f"Error finalizing research session: {e}")
                break

            # Run Fusion Engine (legacy)
            fusion_result = fusion_engine.fuse(last_face_metrics, last_voice_metrics, {})
            analytics_data = analytics_engine.process(face_metrics=last_face_metrics, voice_metrics=last_voice_metrics)

            # Run Research Engine (1 Hz throttled)
            research_data = None
            if research_engine and research_engine.should_step():
                research_data = research_engine.step(
                    face=last_face_metrics,
                    voice=last_voice_metrics,
                    body={},
                    ctx={"ssi_raw": analytics_data.get("ssi_raw", 20), "cii_raw": analytics_data.get("cii_raw", 10)},
                )

            # Time-smoothed alert logic
            if fusion_result["score"] >= 60.0:
                consecutive_high_nervousness += 1
                if consecutive_high_nervousness >= 10:
                    await websocket.send_json({
                        "type": "ALERT",
                        "alert_type": "HIGH_NERVOUSNESS",
                        "score": round(fusion_result["score"], 2),
                        "message": "Persistent nervousness detected. Take a deep breath to calm down."
                    })
                    consecutive_high_nervousness = 0
            else:
                consecutive_high_nervousness = max(0, consecutive_high_nervousness - 1)

            # Send updated metrics
            msg = {
                "type": "METRICS_UPDATE",
                "face": last_face_metrics,
                "voice": last_voice_metrics,
                "fusion": fusion_result,
                "analytics": analytics_data,
            }
            if research_data:
                msg["research"] = research_data
            await websocket.send_json(msg)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
        # Finalize research session on disconnect
        if research_engine and session_id:
            try:
                summary = research_engine.finalize()
                RESEARCH_RESULTS[session_id] = summary
                if session_id in RESEARCH_ENGINES:
                    del RESEARCH_ENGINES[session_id]
                logger.info(f"Research session finalized: {session_id}")
            except Exception as e:
                logger.error(f"Error finalizing research session: {e}")
    except Exception as e:
        logger.error(f"Error in websocket monitor: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# ---------------- API & LLM ---------------- #

from google.genai import types

try:
    llm_client = genai.Client()
    logger.info("GenAI client initialized successfully.")
except Exception as e:
    logger.warning(f"Failed to initialize GenAI client: {e}")
    llm_client = None


@app.post("/next_question")
async def generate_next_question(req: NextQuestionRequest):
    # Hard cap on how many questions the interview may ask (mirrors the frontend).
    # Once reached, no further adaptive questions are generated, removing the
    # bottleneck where the system keeps asking even on skipped answers.
    MAX_QUESTIONS = 8
    if req.question_count >= MAX_QUESTIONS:
        return {"question": "", "fsm_state": "Completed", "reached_limit": True}

    ans = req.previous_answer.lower()
    fsm_state = "Baseline"
    fsm_question = ""

    # Get FSM adaptive question if session exists
    if req.session_id and req.session_id in RESEARCH_ENGINES:
        engine = RESEARCH_ENGINES[req.session_id]
        fsm_state = engine.fsm.state.value
        fsm_question = engine.fsm.ask_question()

    # Check for project/research mentions for LLM deep-dive
    if "project" in ans or "research" in ans or "paper" in ans or "thesis" in ans:
        if llm_client:
            try:
                prompt = (
                    f"The candidate is in an interview for a {req.domain} role. "
                    f"They just explained their project/research/number of projects: '{req.previous_answer}'\n"
                    "Ask them exactly ONE challenging, specific interview question based on the projects "
                    "or topics they just described. Ensure the question is directly relevant to what they mentioned. Return ONLY the question."
                )
                response = await llm_client.aio.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                generated_q = response.text.strip()
                if generated_q:
                    return {"question": generated_q, "fsm_state": fsm_state, "source": "llm_deep_dive"}
            except Exception as e:
                logger.error(f"LLM Error generating dynamic project question: {e}")

        # Fallback if LLM fails
        return {"question": "Could you dive deeper into the methodology or challenges of this project?", "fsm_state": fsm_state, "source": "fallback"}

    # FSM adaptive question (primary research deliverable)
    if fsm_question:
        return {"question": fsm_question, "fsm_state": fsm_state, "source": "fsm_adaptive"}

    # Default: empty to let frontend pick next question
    return {"question": "", "fsm_state": fsm_state}


@app.post("/evaluate_multimodal")
async def evaluate_multimodal(
    answers_json: str = Form(...),
):
    try:
        payload = json.loads(answers_json)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    answers = payload.get("answers", [])
    domain = payload.get("domain", "general")
    session_id = payload.get("session_id")

    per_q_results = []

    total_conf = total_clarity = total_acc = total_nerv = 0.0
    valid_answers = 0
    total_questions = len(answers)

    # ---------------- PER QUESTION SCORING ---------------- #

    for item in answers:
        q = item.get("question", "")
        ans = (item.get("answer") or "").strip()

        tokens = tokenize(ans)
        wc = len(tokens)
        filler = sum(1 for t in tokens if t in FILLER_WORDS)
        filler_freq = filler / wc if wc > 0 else 0

        if wc == 0:
            per_q_results.append({
                "question": q,
                "transcript": "",
                "confidence": 0,
                "clarity": 0,
                "accuracy": 0,
                "nervousness_index": 100,
            })
            continue

        valid_answers += 1

        # Confidence
        if wc < 15:
            confidence = 25 + wc
        elif wc < 40:
            confidence = 45 + (wc - 15) * 1.2
        else:
            confidence = 75 + min(15, (wc - 40) * 0.3)
        confidence -= filler * 3

        # Clarity
        clarity = 85 - filler_freq * 60
        if wc < 12:
            clarity -= 20

        # Accuracy via LLM
        mistakes = ""
        accuracy_score = 50 + min(30, wc * 0.6)
        if llm_client and wc >= 15:
            try:
                sys_prompt = (
                    "You are an expert technical interviewer. Evaluate the candidate's answer for technical accuracy.\n"
                    "Provide a JSON response containing strictly two keys:\n"
                    "- \"accuracy\": an integer from 0 to 100 scoring the technical correctness and relevance.\n"
                    "- \"mistakes\": a concise string pointing out any factual errors or misconceptions. If mostly correct, say 'None'."
                )
                user_prompt = f"Question: {q}\nCandidate Answer: {ans}"
                response = await llm_client.aio.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[sys_prompt, user_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                try:
                    result = json.loads(response.text)
                    if "accuracy" in result:
                        accuracy_score = float(result["accuracy"])
                    if "mistakes" in result:
                        mistakes = str(result["mistakes"])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM evaluation JSON: {response.text}")
            except Exception as e:
                logger.error(f"LLM Error evaluating accuracy: {e}")

        accuracy = accuracy_score - filler_freq * 40
        nervousness = filler_freq * 70 + (20 if wc < 12 else 0)

        confidence = clamp(confidence)
        clarity = clamp(clarity)
        accuracy = clamp(accuracy)
        nervousness = clamp(nervousness)

        per_q_results.append({
            "question": q,
            "transcript": ans,
            "confidence": round(confidence, 2),
            "clarity": round(clarity, 2),
            "accuracy": round(accuracy, 2),
            "nervousness_index": round(nervousness, 2),
            "mistakes": mistakes
        })

        total_conf += confidence
        total_clarity += clarity
        total_acc += accuracy
        total_nerv += nervousness

    # ---------------- NO ANSWERS ---------------- #

    if valid_answers == 0:
        result = {
            "confidence": 0, "clarity": 0, "accuracy": 0, "nervousness_index": 100,
            "overall": 0, "answered": 0, "skipped": total_questions,
            "completion_rate": 0, "feedback": "No answers detected.",
            "detail_per_question": per_q_results, "domain": domain,
        }
        if session_id and session_id in RESEARCH_RESULTS:
            result["research"] = RESEARCH_RESULTS.pop(session_id)
        return result

    # ---------------- AGGREGATION ---------------- #

    avg_conf = total_conf / valid_answers
    avg_clarity = total_clarity / valid_answers
    avg_acc = total_acc / valid_answers
    avg_nerv = total_nerv / valid_answers

    base_overall = (
        avg_conf * 0.35 +
        avg_clarity * 0.30 +
        avg_acc * 0.25 +
        (100 - avg_nerv) * 0.10
    )

    skipped = total_questions - valid_answers
    answer_ratio = valid_answers / total_questions
    skip_penalty = (1 - skipped / total_questions) ** 1.7

    scaled_overall = base_overall * answer_ratio * skip_penalty

    if answer_ratio >= 0.9:
        cap = 100
    elif answer_ratio >= 0.7:
        cap = 85
    elif answer_ratio >= 0.5:
        cap = 65
    else:
        cap = 45

    overall = clamp(min(scaled_overall, cap))

    result = {
        "domain": domain,
        "confidence": round(avg_conf, 2),
        "clarity": round(avg_clarity, 2),
        "accuracy": round(avg_acc, 2),
        "nervousness_index": round(avg_nerv, 2),
        "overall": round(overall, 2),
        "answered": valid_answers,
        "skipped": skipped,
        "completion_rate": round(answer_ratio * 100, 2),
        "feedback": "Score adjusted based on answer quality and completion rate.",
        "detail_per_question": per_q_results,
    }

    # Attach research summary if available
    if session_id:
        if session_id in RESEARCH_ENGINES:
            try:
                summary = RESEARCH_ENGINES[session_id].finalize()
                result["research"] = summary
                RESEARCH_RESULTS[session_id] = summary
                del RESEARCH_ENGINES[session_id]
            except Exception as e:
                logger.error(f"Error finalizing research for evaluation: {e}")
        elif session_id in RESEARCH_RESULTS:
            # Keep the summary in RESEARCH_RESULTS (non-destructive read) so the
            # Results page can still fetch /interview/results/{session_id}.
            result["research"] = RESEARCH_RESULTS[session_id]
            try:
                sess_dir = os.path.join(interview_api.SESSIONS_DIR, session_id)
                os.makedirs(os.path.join(sess_dir, "plots"), exist_ok=True)
                with open(os.path.join(sess_dir, "results.json"), "w", encoding="utf-8") as f:
                    json.dump(RESEARCH_RESULTS[session_id], f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Could not persist interview results to disk: {e}")

    return result


# ---------------- RESEARCH ENDPOINTS ---------------- #

@app.get("/research/sessions")
async def list_research_sessions():
    return {
        "active": list(RESEARCH_ENGINES.keys()),
        "completed": list(RESEARCH_RESULTS.keys()),
    }


@app.get("/research/summary/{session_id}")
async def get_research_summary(session_id: str):
    if session_id in RESEARCH_RESULTS:
        return RESEARCH_RESULTS[session_id]
    if session_id in RESEARCH_ENGINES:
        return RESEARCH_ENGINES[session_id].finalize()
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/sessions/{session_id}/research-summary")
async def get_session_research_summary(session_id: str):
    """Expose the full InterviewEngine.finalize() summary for a completed session."""
    # 1. Check persisted results first (already finalized)
    if session_id in RESEARCH_RESULTS:
        return RESEARCH_RESULTS[session_id]

    # 2. If engine is still in memory but not yet finalized, finalize now and cache
    if session_id in RESEARCH_ENGINES:
        engine = RESEARCH_ENGINES[session_id]
        summary = engine.finalize()
        RESEARCH_RESULTS[session_id] = summary
        del RESEARCH_ENGINES[session_id]
        return summary

    # 3. Not found
    raise HTTPException(status_code=404, detail="Session not found")


# ---------------- TRAINING & PARTICIPANT ENDPOINTS ---------------- #

@app.post("/train")
async def train_models():
    """Train (calibrate) models on all participant datasets in the backend directory."""
    global CALIBRATION_DATA
    try:
        from training.pipeline import run_training
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(backend_dir, "trained_models")
        report = run_training(backend_dir, output_dir)
        CALIBRATION_DATA = report
        return {
            "status": "success",
            "message": f"Models calibrated on {report.get('training_summary', {}).get('n_participants', 0)} participants",
            "summary": report.get("training_summary", {}),
            "bilstm": report.get("bilstm_calibration", {}),
            "au_cnn": report.get("au_cnn_calibration", {}),
            "phq8": report.get("phq8_calibration", {}),
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/train/status")
async def training_status():
    """Check if models have been calibrated."""
    return {
        "calibrated": CALIBRATION_DATA is not None,
        "training_dir": TRAINING_OUTPUT_DIR,
        "report_exists": os.path.exists(os.path.join(TRAINING_OUTPUT_DIR, "calibration_report.json")),
        "summary": CALIBRATION_DATA.get("training_summary", {}) if CALIBRATION_DATA else None,
    }


@app.get("/train/reload")
async def reload_calibration():
    """Reload calibration data from disk."""
    global CALIBRATION_DATA
    _load_calibration()
    return {
        "calibrated": CALIBRATION_DATA is not None,
        "summary": CALIBRATION_DATA.get("training_summary", {}) if CALIBRATION_DATA else None,
    }


@app.get("/participants")
async def list_participants():
    """List all participant directories found in the backend."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    participants = []
    for entry in sorted(os.listdir(backend_dir)):
        full = os.path.join(backend_dir, entry)
        if os.path.isdir(full) and entry.endswith("_P"):
            import glob
            csvs = glob.glob(os.path.join(full, "*.csv")) + glob.glob(os.path.join(full, "*.txt"))
            participants.append({
                "id": entry,
                "path": full,
                "files": [os.path.basename(f) for f in csvs],
                "file_count": len(csvs),
            })
    return {"participants": participants, "count": len(participants)}


@app.post("/analyze_participant/{participant_id}")
async def analyze_participant(participant_id: str):
    """Run the full research pipeline on a specific participant's data."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    participant_dir = os.path.join(backend_dir, participant_id)

    if not os.path.isdir(participant_dir):
        raise HTTPException(status_code=404, detail=f"Participant {participant_id} not found")

    try:
        from research.offline_analysis import run_offline_from_participant_dir
        summary = run_offline_from_participant_dir(participant_dir)
        return summary
    except Exception as e:
        logger.error(f"Participant analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze_all_participants")
async def analyze_all_participants():
    """Run analysis on ALL participant directories and return aggregated results."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        from research.offline_analysis import run_all_participants
        summary = run_all_participants(backend_dir)
        return summary
    except Exception as e:
        logger.error(f"Multi-participant analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------------- PLOTS ---------------- #

import glob as _glob
from fastapi.responses import FileResponse, JSONResponse

PLOTS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_plots")
AGG_PLOT_PID = "_all"


@app.post("/analysis/_all/generate_plots")
async def generate_all_participant_plots(domain: str = "general"):
    """Run aggregated analysis on ALL participants and generate cross-participant
    plots: FSM heatmap + PHQ-8 score bar chart."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(PLOTS_OUTPUT_DIR, AGG_PLOT_PID)
    os.makedirs(plot_dir, exist_ok=True)

    try:
        from research.offline_analysis import (
            run_all_participants,
            plot_fsm_heatmap,
            plot_phq8_scores,
        )
        summary = run_all_participants(backend_dir, domain)

        if "error" in summary:
            raise HTTPException(status_code=500, detail=str(summary["error"]))

        fig_paths = []
        p = plot_fsm_heatmap(summary, plot_dir)
        if p:
            fig_paths.append(p)
        p = plot_phq8_scores(summary, plot_dir)
        if p:
            fig_paths.append(p)

        plots = [
            {
                "name": os.path.basename(fp),
                "path": fp,
                "url": f"/analysis/_all/plots/{os.path.basename(fp)}",
            }
            for fp in fig_paths
        ]
        return {
            "participant_id": AGG_PLOT_PID,
            "plots": plots,
            "population": summary.get("population", {}),
            "participant_ids": summary.get("participant_ids", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"All-participants plot generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")


@app.get("/analysis/_all/plots")
async def list_all_participant_plots():
    """List available cross-participant plot files."""
    plot_dir = os.path.join(PLOTS_OUTPUT_DIR, AGG_PLOT_PID)
    if not os.path.isdir(plot_dir):
        return {"participant_id": AGG_PLOT_PID, "plots": []}

    plots = []
    for f in sorted(os.listdir(plot_dir)):
        if f.endswith(".png"):
            plots.append({
                "name": f,
                "url": f"/analysis/_all/plots/{f}",
            })
    return {"participant_id": AGG_PLOT_PID, "plots": plots}


@app.get("/analysis/{participant_id}/generate_plots")
async def generate_participant_plots(participant_id: str, domain: str = "general"):
    """Generate and save plots for a specific participant. Returns list of plot files."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    participant_dir = os.path.join(backend_dir, participant_id)

    if not os.path.isdir(participant_dir):
        raise HTTPException(status_code=404, detail=f"Participant {participant_id} not found")

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(PLOTS_OUTPUT_DIR, participant_id)
    os.makedirs(plot_dir, exist_ok=True)

    try:
        from research.offline_analysis import run_offline_from_participant_dir, plot_summary
        summary = run_offline_from_participant_dir(participant_dir, domain)
        fig_paths = plot_summary(summary, plot_dir)

        plots = []
        for fp in fig_paths:
            plots.append({
                "name": os.path.basename(fp),
                "path": fp,
                "url": f"/analysis/{participant_id}/plots/{os.path.basename(fp)}",
            })
        return {"participant_id": participant_id, "plots": plots}
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {str(e)}")


@app.get("/analysis/{participant_id}/plots")
async def list_participant_plots(participant_id: str):
    """List available plot files for a participant."""
    plot_dir = os.path.join(PLOTS_OUTPUT_DIR, participant_id)
    if not os.path.isdir(plot_dir):
        return {"participant_id": participant_id, "plots": []}

    plots = []
    for f in sorted(os.listdir(plot_dir)):
        if f.endswith(".png"):
            plots.append({
                "name": f,
                "url": f"/analysis/{participant_id}/plots/{f}",
            })
    return {"participant_id": participant_id, "plots": plots}


@app.get("/analysis/{participant_id}/plots/{filename}")
async def serve_participant_plot(participant_id: str, filename: str):
    """Serve a participant plot image."""
    if not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG plots are served")
    plot_dir = os.path.join(PLOTS_OUTPUT_DIR, participant_id)
    file_path = os.path.join(plot_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path, media_type="image/png")


# ---------------- INTERVIEW REST API (Section 7) ---------------- #

class InterviewStartRequest(BaseModel):
    domain: str = "general"
    candidate_name: str = ""


class InterviewNextRequest(BaseModel):
    session_id: str
    answer: str = ""


class InterviewEndRequest(BaseModel):
    session_id: str
    recording_path: Optional[str] = None
    transcript: Optional[str] = None


@app.post("/interview/start")
async def interview_start(req: InterviewStartRequest):
    """Start a new interview session. Returns session_id + max questions."""
    return INTERVIEW_MGR.start(domain=req.domain, candidate_name=req.candidate_name)


@app.post("/interview/next_question")
async def interview_next(req: InterviewNextRequest):
    """FSM-adaptive next question. Marks is_final at N_MAX_QUESTIONS=8."""
    res = INTERVIEW_MGR.next_question(req.session_id, answer=req.answer)
    if res.get("status_code") == 404:
        raise HTTPException(status_code=404, detail="Session not found")
    return res


@app.post("/interview/end")
async def interview_end(req: InterviewEndRequest):
    """End the interview; persist recording/transcript and launch analysis job."""
    if req.transcript:
        meta = interview_api.SESSION_META.get(req.session_id)
        if meta:
            with open(os.path.join(meta["session_dir"], "transcript.txt"), "w",
                      encoding="utf-8") as f:
                f.write(req.transcript)
    res = INTERVIEW_MGR.end(req.session_id, recording=req.recording_path)
    if res.get("status_code") == 404:
        raise HTTPException(status_code=404, detail="Session not found")
    return res


@app.get("/interview/status/{job_id}")
async def interview_job_status(job_id: str):
    """Poll an analysis job's status."""
    job = interview_api.JOB_STATE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    out = {"job_id": job_id, "status": job.get("status"),
           "session_id": job.get("session_id")}
    if job.get("status") == "failed":
        out["error"] = job.get("error")
    return out


@app.get("/interview/results/{session_id}")
async def interview_results(session_id: str, job_id: Optional[str] = None):
    """Return full results (plots + metrics + recommendation) for a session.

    Priority:
      1. A completed analysis job (via /interview/end).
      2. The live WebSocket-accumulated engine summary for this session
         (plots + recommendation generated on demand from real live data).
    """
    # 1. Completed job from the interview manager
    res = INTERVIEW_MGR.results(session_id, job_id=job_id)
    if res.get("status_code") == 404:
        # Session unknown to the manager — may be a live WS-only session.
        manager_known = False
        res = None
    elif res.get("status") == "completed":
        return res
    else:
        manager_known = True

    # 2. WS-accumulated live summary (live WS sessions are not in SESSION_META)
    summary = None
    if session_id in RESEARCH_RESULTS:
        summary = RESEARCH_RESULTS[session_id]
    elif session_id in RESEARCH_ENGINES:
        try:
            summary = RESEARCH_ENGINES[session_id].finalize()
            RESEARCH_RESULTS[session_id] = summary
            del RESEARCH_ENGINES[session_id]
        except Exception as e:
            logger.error(f"Failed to finalize live engine: {e}")

    if summary is None:
        if not manager_known:
            raise HTTPException(status_code=404, detail="Session not found")
        if isinstance(res, dict) and isinstance(res.get("status"), str):
            return res  # still processing / pending
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = os.path.join(interview_api.SESSIONS_DIR, session_id)
    os.makedirs(os.path.join(session_dir, "plots"), exist_ok=True)
    from research.report_plots import generate_report_plots
    from research.recommendation import generate_recommendation
    plots = generate_report_plots(summary, os.path.join(session_dir, "plots"))
    report = generate_recommendation(summary)
    return {
        "session_id": session_id,
        "status": "completed",
        "metrics": interview_api._public_metrics(summary),
        "plots": {
            k: f"/interview/session/plot/{session_id}/{os.path.basename(v)}"
            for k, v in plots.items()
        },
        "recommendation": report,
        "transcript_features": summary.get("transcript_features"),
    }


@app.get("/interview/session/plot/{session_id}/{filename}")
async def serve_session_plot(session_id: str, filename: str):
    """Serve a session plot PNG."""
    if not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG plots are served")
    meta = interview_api.SESSION_META.get(session_id)
    if meta:
        file_path = os.path.join(meta["session_dir"], "plots", filename)
    else:
        # WS-only sessions may not have meta; look up the session dir directly
        candidate = os.path.join(interview_api.SESSIONS_DIR, session_id, "plots", filename)
        file_path = candidate if os.path.exists(candidate) else None
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path, media_type="image/png")


@app.get("/interview/download/{session_id}")
async def interview_download(session_id: str):
    """Download a ZIP of all session artifacts."""
    zip_path = INTERVIEW_MGR.download_zip(session_id)
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Session or archive not found")
    return FileResponse(zip_path, media_type="application/zip",
                        filename=f"{session_id}_results.zip")


# ---------------- RESEARCH FIGURE ENDPOINTS ---------------- #


def _resolve_question_count(session_id: str) -> int:
    """Return the number of questions answered for a session.

    Precedence: in-memory SESSION_META > persisted meta.json. The count is
    incremented in ``InterviewSessionManager.next_question`` and saved to
    ``meta.json`` via ``_save_meta``, so it survives restarts. Falls back
    to 0 when no source is available.
    """
    meta = interview_api.SESSION_META.get(session_id) or {}
    qc = meta.get("question_count") or meta.get("n_questions")
    if qc:
        return int(qc)

    meta_path = os.path.join(interview_api.SESSIONS_DIR, session_id, "meta.json")
    if not os.path.exists(meta_path):
        return 0
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            disk_meta = json.load(f)
        qc = disk_meta.get("question_count") or disk_meta.get("n_questions")
        return int(qc) if qc else 0
    except Exception:
        return 0

@app.get("/results/research-figure/{session_id}")
async def research_figure_html(session_id: str):
    """Render a 1400x900 research figure as a self-contained HTML page."""
    summary = None

    # 1. Check in-memory finalized results
    if session_id in RESEARCH_RESULTS:
        summary = RESEARCH_RESULTS[session_id]

    # 2. Check live engine — finalize on demand
    if summary is None and session_id in RESEARCH_ENGINES:
        try:
            summary = RESEARCH_ENGINES[session_id].finalize()
            RESEARCH_RESULTS[session_id] = summary
            del RESEARCH_ENGINES[session_id]
        except Exception as e:
            logger.error(f"Failed to finalize live engine for research figure: {e}")

    # 3. Check disk-persisted results.json
    if summary is None:
        summary_path = os.path.join(interview_api.SESSIONS_DIR, session_id, "results.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results.json for {session_id}: {e}")

    if summary is None:
        from fastapi.responses import HTMLResponse as _HTMLResp
        return _HTMLResp(
            content=f"<html><body><h2>Session not found</h2><p>No results for <code>{session_id}</code>.</p></body></html>",
            status_code=404,
        )

    # Inject n_questions from SESSION_META / persisted meta.json
    summary["n_questions"] = _resolve_question_count(session_id)

    from fastapi.responses import HTMLResponse
    from research.research_figure import generate_research_figure
    html = generate_research_figure(summary)
    return HTMLResponse(content=html, status_code=200)


@app.get("/results/research-figure/{session_id}/download")
async def research_figure_download(session_id: str):
    """Download the research figure as a PNG screenshot via Playwright."""
    summary = None
    if session_id in RESEARCH_RESULTS:
        summary = RESEARCH_RESULTS[session_id]

    if summary is None and session_id in RESEARCH_ENGINES:
        try:
            summary = RESEARCH_ENGINES[session_id].finalize()
            RESEARCH_RESULTS[session_id] = summary
            del RESEARCH_ENGINES[session_id]
        except Exception as e:
            logger.error(f"Failed to finalize engine for download: {e}")

    if summary is None:
        summary_path = os.path.join(interview_api.SESSIONS_DIR, session_id, "results.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                pass

    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Inject n_questions from SESSION_META / persisted meta.json
    summary["n_questions"] = _resolve_question_count(session_id)

    from research.research_figure import generate_research_figure
    html = generate_research_figure(summary)
    sid_short = session_id[:8]

    # Try Playwright screenshot
    try:
        import asyncio

        def _screenshot(html_content: str):
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1400, "height": 900})
                page.set_content(html_content, wait_until="load")
                page.wait_for_timeout(2000)
                screenshot = page.screenshot(full_page=False, type="png")
                browser.close()
                return screenshot

        png_bytes = await asyncio.to_thread(_screenshot, html)

        from fastapi.responses import Response
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="research_figure_{sid_short}.png"'
            },
        )
    except Exception as e:
        logger.error(f"Playwright screenshot failed: {e}", exc_info=True)
        from fastapi.responses import HTMLResponse
        note = (
            '<div style="background:#fef3c7;border:1px solid #f59e0b;padding:12px;'
            'margin:12px;font-family:sans-serif;font-size:13px;">'
            f"<b>Screenshot failed ({type(e).__name__}):</b> {e}. "
            "Use browser screenshot instead.</div>"
        )
        html_with_note = html.replace("<body>", f"<body>{note}", 1)
        return HTMLResponse(content=html_with_note, status_code=200)


@app.get("/interview/n_max_questions")
async def interview_n_max_questions():
    return {"n_max_questions": N_MAX_QUESTIONS}


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
