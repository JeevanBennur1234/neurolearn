"""
api/main.py — NeuroLearn-GEN v2 FastAPI Application
Complete adaptive learning system API.
"""
import os, uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="NeuroLearn-GEN v2",
    description="Adaptive Learning System: Teach → Quiz → Review → Reteach → Test",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class StartTopicRequest(BaseModel):
    topic: str = Field(..., min_length=2, description="Topic to learn (any subject)")
    session_id: Optional[str] = None

class QuizAnswersRequest(BaseModel):
    session_id: str
    answers: dict[str, str] = Field(..., description='{"0":"A","1":"C",...}')

class SessionRequest(BaseModel):
    session_id: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {"system": "NeuroLearn-GEN v2", "docs": "/docs",
            "flow": "POST /learn/start → /learn/quiz/start → /learn/quiz/submit → /learn/reteach → /learn/test/start → /learn/test/submit"}


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "provider": os.getenv("LLM_PROVIDER", "gemini"), "version": "2.0.0"}


# ── LEARNING FLOW ─────────────────────────────────────────────────────────────

@app.post("/learn/start", tags=["Learning Flow"])
async def start_topic(req: StartTopicRequest):
    """
    STEP 1 — Start learning a topic.
    Extracts subtopics → generates comprehensive lesson → returns lesson content.
    Response includes: full lesson text + next_action = 'show_quiz_prompt'
    """
    from core.learning_orchestrator import get_orchestrator
    session_id = req.session_id or str(uuid.uuid4())
    try:
        result = get_orchestrator().start_topic(session_id, req.topic)
        result["session_id"] = session_id
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/learn/quiz/start", tags=["Learning Flow"])
async def start_quiz(req: SessionRequest):
    """
    STEP 2 — Start mini-quiz after lesson.
    Returns 5 MCQ questions (no answers). Frontend collects user selections.
    Response includes: questions[] array + next_action = 'collect_answers'
    """
    from core.learning_orchestrator import get_orchestrator
    try:
        return get_orchestrator().start_mini_quiz(req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/learn/quiz/submit", tags=["Learning Flow"])
async def submit_quiz(req: QuizAnswersRequest):
    """
    STEP 3 — Submit mini-quiz answers.
    Evaluates answers, identifies weak areas, generates detailed review.
    Response includes: review text + weak_areas[] + next_action:
      - 'offer_reteach' if weak areas found
      - 'offer_full_test' if performance is good
    """
    from core.learning_orchestrator import get_orchestrator
    try:
        return get_orchestrator().submit_quiz_answers(req.session_id, req.answers)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/learn/reteach", tags=["Learning Flow"])
async def reteach(req: SessionRequest):
    """
    STEP 4 (conditional) — Re-teach weak areas.
    Uses a fresh angle and new analogies. Returns reteach lesson.
    After this, frontend should call /learn/quiz/start again to loop.
    next_action = 'show_quiz_prompt'
    """
    from core.learning_orchestrator import get_orchestrator
    try:
        return get_orchestrator().reteach(req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/learn/test/start", tags=["Learning Flow"])
async def start_full_test(req: SessionRequest):
    """
    STEP 5 — User says they're ready. Start the formal 10-question test.
    Returns 10 MCQ questions. next_action = 'collect_final_answers'
    """
    from core.learning_orchestrator import get_orchestrator
    try:
        return get_orchestrator().start_full_test(req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/learn/test/submit", tags=["Learning Flow"])
async def submit_final_test(req: QuizAnswersRequest):
    """
    STEP 6 — Submit final test answers.
    Evaluates, scores, generates comprehensive final report.
    Response includes: full report + final_score + next_action = 'show_report'
    """
    from core.learning_orchestrator import get_orchestrator
    try:
        return get_orchestrator().submit_final_test(req.session_id, req.answers)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/learn/reset", tags=["Learning Flow"])
async def reset_session(req: SessionRequest):
    """Reset a session to start a new topic."""
    from core.learning_orchestrator import get_orchestrator
    return get_orchestrator().reset(req.session_id)


@app.get("/learn/session/{session_id}", tags=["Learning Flow"])
async def get_session_status(session_id: str):
    """Get current session state (phase, score, weak areas, etc.)"""
    from core.learning_session import get_session
    try:
        s = get_session(session_id)
        return {
            "session_id": session_id,
            "topic": s.topic,
            "phase": s.phase,
            "quiz_rounds": s.quiz_rounds,
            "reteach_rounds": s.reteach_rounds,
            "overall_score": round(s.get_overall_score() * 100, 1),
            "weak_areas": s.weak_areas,
            "subtopics": s.subtopics,
            "final_score": round((s.final_score or 0) * 100, 1),
        }
    except Exception as e:
        raise HTTPException(500, str(e))
