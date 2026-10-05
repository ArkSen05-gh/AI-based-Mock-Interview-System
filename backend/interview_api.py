"""
Interview REST API (Section 7) — REST endpoints for starting/advancing/ending an
interview, running the post-interview offline extraction + 4-plot generation +
LLM (Groq) recommendation, and serving/downloading the results.

The WebSocket live flow is unchanged; these endpoints complement it for the
non-realtime client flow and the post-interview results page.
"""
import os
import json
import time
import uuid
import shutil
import threading
import logging
from typing import Dict, Optional

logger = logging.getLogger("interview_api")

from research.fsm_controller import N_MAX_QUESTIONS
from research.interview_engine import InterviewEngine
from research.session_analyzer import analyze_session
from research.report_plots import generate_report_plots
from research.recommendation import generate_recommendation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
PLOTS_DIR = os.path.join(BASE_DIR, "sessions", "_plots")


def _ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


_ensure_dirs()

# in-memory job state
JOB_STATE: Dict[str, Dict] = {}
SESSION_META: Dict[str, Dict] = {}


class InterviewSessionManager:
    """Stores interview sessions and analysis job results in memory/disk."""

    def __init__(self, calibration=None):
        self.calibration = calibration
        _ensure_dirs()

    # ------------------------------------------------------------------
    def start(self, domain: str = "general", candidate_name: str = "") -> Dict:
        session_id = f"int_{uuid.uuid4().hex[:12]}"
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        engine = InterviewEngine(domain=domain, session_id=session_id,
                                 calibration=self.calibration)
        SESSION_META[session_id] = {
            "domain": domain,
            "candidate_name": candidate_name,
            "started_at": time.time(),
            "question_count": 0,
            "fsm_state": "Baseline",
            "history": [],
            "session_dir": session_dir,
            "status": "in_progress",
        }
        self._save_meta(session_id)
        return {
            "session_id": session_id,
            "fsm_state": "Baseline",
            "n_max_questions": N_MAX_QUESTIONS,
            "interviewer_start": _warmup_question(domain),
        }

    # ------------------------------------------------------------------
    def next_question(self, session_id: str, answer: str = "") -> Dict:
        meta = SESSION_META.get(session_id)
        if not meta:
            return {"error": "Session not found", "status_code": 404}
        if meta.get("question_count", 0) >= N_MAX_QUESTIONS:
            return {
                "question": "",
                "fsm_state": "Completed",
                "is_final": True,
                "reached_limit": True,
                "question_count": meta["question_count"],
            }
        engine = RESEARCH_ENGINES.get(session_id)
        if engine is None:
            engine = InterviewEngine(domain=meta.get("domain", "general"),
                                     session_id=session_id, calibration=self.calibration)
            RESEARCH_ENGINES[session_id] = engine
        meta["question_count"] += 1
        q = engine.fsm.ask_question()
        meta["history"].append({
            "question": q,
            "answer": answer,
            "fsm_state": engine.fsm.state.value,
        })
        meta["fsm_state"] = engine.fsm.state.value
        meta["question_count"] = meta["question_count"]
        self._save_meta(session_id)
        return {
            "question": q,
            "fsm_state": meta["fsm_state"],
            "is_final": meta["question_count"] >= N_MAX_QUESTIONS,
            "question_count": meta["question_count"],
            "n_max_questions": N_MAX_QUESTIONS,
            "source": "fsm_adaptive",
        }

    # ------------------------------------------------------------------
    def end(self, session_id: str, recording: Optional[str] = None) -> Dict:
        meta = SESSION_META.get(session_id)
        if not meta:
            return {"error": "Session not found", "status_code": 404}
        meta["status"] = "processing"
        meta["ended_at"] = time.time()
        if recording:
            # copy the recording into the session dir as recording.mp4 if absent
            sess_dir = meta["session_dir"]
            if not os.path.exists(os.path.join(sess_dir, "recording.mp4")):
                try:
                    ext = os.path.splitext(recording)[1] or ".webm"
                    shutil.copy(recording, os.path.join(sess_dir, "recording" + ext))
                except Exception as e:
                    logger.warning(f"Failed to stage recording: {e}")
        self._save_meta(session_id)

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        JOB_STATE[job_id] = {"status": "queued", "session_id": session_id,
                              "created_at": time.time()}
        threading.Thread(
            target=self._run_job, args=(job_id, session_id, meta),
            daemon=True,
        ).start()
        return {"job_id": job_id, "session_id": session_id, "status": "queued"}

    # ------------------------------------------------------------------
    def _run_job(self, job_id: str, session_id: str, meta: Dict):
        JOB_STATE[job_id]["status"] = "running"
        session_dir = meta["session_dir"]
        try:
            summary = analyze_session(
                session_id, sessions_dir=SESSIONS_DIR,
                domain=meta.get("domain", "general"),
                calibration=self.calibration,
            )
            plots = generate_report_plots(
                summary, os.path.join(session_dir, "plots"))
            report = generate_recommendation(summary)
            result = {
                "session_id": session_id,
                "status": "completed",
                "metrics": _public_metrics(summary),
                "plots": {k: self._serve_url(v) for k, v in plots.items()},
                "recommendation": report,
                "transcript_features": summary.get("transcript_features"),
            }
            meta["status"] = "completed"
            self._save_meta(session_id)
            JOB_STATE[job_id]["result"] = result
            JOB_STATE[job_id]["status"] = "completed"
            logger.info(f"Analysis job {job_id} completed for {session_id}")
        except Exception as e:
            logger.exception(f"Analysis job {job_id} failed: {e}")
            JOB_STATE[job_id]["status"] = "failed"
            JOB_STATE[job_id]["error"] = str(e)

    # ------------------------------------------------------------------
    def results(self, session_id: str, job_id: Optional[str] = None) -> Dict:
        meta = SESSION_META.get(session_id)
        if not meta:
            # WS-only sessions are not in SESSION_META; check persisted results
            return self._results_from_disk(session_id)
        if meta.get("status") != "completed":
            return {
                "session_id": session_id,
                "status": meta.get("status", "processing"),
            }
        if job_id and job_id in JOB_STATE and JOB_STATE[job_id].get("result"):
            return JOB_STATE[job_id]["result"]
        # rebuild from saved summary/plots if job state lost (server restart)
        return self._results_from_disk(session_id)

    def _results_from_disk(self, session_id: str) -> Dict:
        summary_path = os.path.join(SESSIONS_DIR, session_id, "results.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                plots = generate_report_plots(
                    summary, os.path.join(SESSIONS_DIR, session_id, "plots"))
                return {
                    "session_id": session_id,
                    "status": "completed",
                    "metrics": _public_metrics(summary),
                    "plots": {k: self._serve_url(v) for k, v in plots.items()},
                    "recommendation": generate_recommendation(summary),
                }
            except Exception as e:
                logger.warning(f"Failed to rebuild results from disk for {session_id}: {e}")
                return {"error": "Session not found", "status_code": 404}
        return {"error": "Session not found", "status_code": 404}

    # ------------------------------------------------------------------
    def download_zip(self, session_id: str) -> Optional[str]:
        meta = SESSION_META.get(session_id)
        if not meta:
            return None
        src = meta["session_dir"]
        zip_path = os.path.join(PLOTS_DIR, f"{session_id}.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(zip_path[:-4], "zip", src)
        return zip_path

    # ------------------------------------------------------------------
    def _save_meta(self, session_id: str):
        meta = SESSION_META.get(session_id)
        if not meta:
            return
        try:
            with open(os.path.join(meta["session_dir"], "meta.json"), "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in meta.items() if k != "history"},
                          f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save meta: {e}")

    @staticmethod
    def _serve_url(path: str) -> str:
        return f"/interview/session/plot/{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"

    def restore_from_disk(self):
        for name in os.listdir(SESSIONS_DIR):
            meta_path = os.path.join(SESSIONS_DIR, name, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                m["session_dir"] = os.path.join(SESSIONS_DIR, name)
                SESSION_META[m.get("session_id", name)] = m
            except Exception:
                continue


def _public_metrics(summary: Dict) -> Dict:
    return {
        "duration_seconds": summary.get("duration_seconds"),
        "n_frames": summary.get("n_frames"),
        "stress_mean": summary.get("stress_mean"),
        "stress_max": summary.get("stress_max"),
        "confidence_mean": summary.get("confidence_mean"),
        "pearson_r_au_cnn": summary.get("pearson_r_au_cnn"),
        "phq8": (summary.get("phq8") or {}).get("phq8_score",
                                                summary.get("phq8_score")),
        "phq8_detail": summary.get("phq8"),
        "fsm_final_state": (summary.get("fsm_session") or {}).get("final_state"),
        "fsm_state_pct": summary.get("fsm_state_pct"),
        "n_transitions": (summary.get("fsm_session") or {}).get("n_transitions"),
        "stress_curve_100": summary.get("stress_curve_100"),
        "confidence_curve_100": summary.get("confidence_curve_100"),
    }


# module-level engine registry (mirrors app.py) for next_question
RESEARCH_ENGINES: Dict[str, InterviewEngine] = {}


def _warmup_question(domain: str) -> str:
    return ("Please share a brief introduction about yourself and your "
            "professional background.")


def build_manager(calibration):
    mgr = InterviewSessionManager(calibration=calibration)
    mgr.restore_from_disk()
    return mgr
