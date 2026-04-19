"""
core/learning_orchestrator.py

THE BRAIN of NeuroLearn v2.
Drives the complete adaptive learning loop:

  start_topic(topic)
      ↓
  [TEACHING]     → TeacherAgent.teach()
      ↓
  [MINI_QUIZ]    → QuizAgent.generate_mini_quiz() → user answers
      ↓
  [REVIEWING]    → ReviewerAgent.review()
      ↓
  [RETEACHING?]  → If weak areas: TeacherAgent.reteach() → back to MINI_QUIZ
      ↓  (loop until user says ready OR score ≥ threshold)
  [READY]        → User triggers
      ↓
  [FULL_TEST]    → QuizAgent.generate_full_test() → user answers
      ↓
  [COMPLETE]     → ReportAgent.generate_report()

API contract — every method returns:
  {
    "phase": str,              # current phase after this action
    "content": str,            # primary text response to show user
    "questions": list | None,  # quiz questions if phase is quiz/test
    "session": dict,           # lightweight session summary
    "next_action": str,        # what the frontend should show next
  }
"""
from __future__ import annotations
from core.learning_session import (
    LearningSession, LearningPhase, QuizAttempt,
    get_session, clear_session,
)
from agents.teacher_agent import TeacherAgent
from agents.quiz_agent import QuizAgent
from agents.reviewer_agent import ReviewerAgent
from agents.report_agent import ReportAgent
from utils.topic_analyzer import extract_subtopics

try:
    from rag.retriever import get_retriever
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False


def _rag(query: str) -> str:
    if not RAG_AVAILABLE:
        return ""
    try:
        return get_retriever().retrieve(query, k=3)
    except Exception:
        return ""


def _session_summary(s: LearningSession) -> dict:
    return {
        "session_id": s.session_id,
        "topic": s.topic,
        "phase": s.phase,
        "quiz_rounds": s.quiz_rounds,
        "overall_score": round(s.get_overall_score() * 100, 1),
        "weak_areas": s.weak_areas,
        "subtopics": s.subtopics,
    }


class LearningOrchestrator:

    # ─── 1. Start a topic ────────────────────────────────────────────────────
    def start_topic(self, session_id: str, topic: str) -> dict:
        """Begin a new learning session: extract subtopics → teach."""
        session = get_session(session_id)
        session.topic = topic.strip()
        session.advance_phase(LearningPhase.TEACHING)

        # Extract subtopics for targeted quiz generation
        subtopics = extract_subtopics(topic)
        session.subtopics = subtopics

        # Teach
        rag_ctx = _rag(topic)
        teacher = TeacherAgent()
        lesson = teacher.teach(topic, rag_ctx)
        session.add_lesson(lesson)

        return {
            "phase": session.phase,
            "content": lesson,
            "questions": None,
            "session": _session_summary(session),
            "next_action": "show_quiz_prompt",  # Ask if ready for mini-quiz
        }

    # ─── 2. Start mini-quiz ──────────────────────────────────────────────────
    def start_mini_quiz(self, session_id: str) -> dict:
        """Generate and return mini-quiz questions (no answers yet)."""
        session = get_session(session_id)
        session.advance_phase(LearningPhase.MINI_QUIZ)
        session.start_new_quiz_batch()

        quiz_agent = QuizAgent()
        rag_ctx = _rag(session.topic)

        # More questions if this is a reteach round
        count = 5 if session.reteach_rounds == 0 else 4

        questions = quiz_agent.generate_mini_quiz(
            topic=session.topic,
            subtopics=session.weak_areas if session.weak_areas else session.subtopics,
            session_context=session.get_context_summary(),
            rag_context=rag_ctx,
            count=count,
        )

        # Store questions on session for answer evaluation later
        session._pending_quiz = questions  # type: ignore[attr-defined]

        return {
            "phase": session.phase,
            "content": f"**Quiz Time! Round {session.quiz_rounds}**\nAnswer all {len(questions)} questions below.",
            "questions": questions,
            "session": _session_summary(session),
            "next_action": "collect_answers",
        }

    # ─── 3. Submit quiz answers ──────────────────────────────────────────────
    def submit_quiz_answers(self, session_id: str, answers: dict[str, str]) -> dict:
        """
        answers = {"0": "A", "1": "C", ...}  (question index → chosen option)
        Evaluates answers, records attempts, triggers review.
        """
        session = get_session(session_id)
        questions = getattr(session, "_pending_quiz", [])

        for idx, q in enumerate(questions):
            user_ans = answers.get(str(idx), "").upper().strip()
            correct = q.get("correct", "").upper().strip()
            attempt = QuizAttempt(
                question=q["question"],
                options=q.get("options", {}),
                correct_answer=correct,
                user_answer=user_ans,
                is_correct=(user_ans == correct),
                explanation=q.get("explanation", ""),
                subtopic=q.get("subtopic", session.topic),
            )
            session.record_quiz_answer(attempt)

        session.advance_phase(LearningPhase.REVIEWING)
        session.compute_weak_areas()

        # Generate review
        reviewer = ReviewerAgent()
        review_text = reviewer.review(
            topic=session.topic,
            attempts=session.current_quiz_batch,
            round_num=session.quiz_rounds,
            session_context=session.get_context_summary(),
        )

        # Decide next phase
        score = session.get_overall_score()
        has_weak = bool(session.weak_areas)

        if has_weak:
            next_action = "offer_reteach"
            next_phase_hint = LearningPhase.RETEACHING
        else:
            next_action = "offer_full_test"
            next_phase_hint = LearningPhase.READY

        return {
            "phase": session.phase,
            "content": review_text,
            "questions": None,
            "session": _session_summary(session),
            "next_action": next_action,
            "score": round(score * 100, 1),
            "weak_areas": session.weak_areas,
        }

    # ─── 4. Reteach weak areas ───────────────────────────────────────────────
    def reteach(self, session_id: str) -> dict:
        """Re-explain weak areas with a fresh approach, then loop to quiz."""
        session = get_session(session_id)
        session.advance_phase(LearningPhase.RETEACHING)
        session.reteach_rounds += 1

        # Build mistakes context for the reteach agent
        wrong_attempts = [a for a in session.current_quiz_batch if not a.is_correct]
        mistakes_ctx = "\n".join(
            f"- Got wrong: '{a.question}' | Their answer: {a.user_answer} | Correct: {a.correct_answer}"
            for a in wrong_attempts
        )

        rag_ctx = _rag(" ".join(session.weak_areas))
        teacher = TeacherAgent()
        reteach_content = teacher.reteach(
            topic=session.topic,
            weak_areas=session.weak_areas,
            previous_lesson="\n".join(session.lesson_content),
            mistakes_context=mistakes_ctx,
            rag_context=rag_ctx,
        )
        session.add_lesson(reteach_content)

        return {
            "phase": session.phase,
            "content": reteach_content,
            "questions": None,
            "session": _session_summary(session),
            "next_action": "show_quiz_prompt",  # Loop back to quiz
        }

    # ─── 5. User declares readiness → Full test ──────────────────────────────
    def start_full_test(self, session_id: str) -> dict:
        """User says they're ready. Generate and return the full formal test."""
        session = get_session(session_id)
        session.advance_phase(LearningPhase.FULL_TEST)

        quiz_agent = QuizAgent()
        rag_ctx = _rag(session.topic)

        questions = quiz_agent.generate_full_test(
            topic=session.topic,
            subtopics=session.subtopics,
            weak_areas=session.weak_areas,
            rag_context=rag_ctx,
            count=10,
        )
        session._pending_quiz = questions  # type: ignore[attr-defined]

        return {
            "phase": session.phase,
            "content": f"## 🎓 Final Test — {session.topic}\n\n10 questions. Take your time. Good luck!",
            "questions": questions,
            "session": _session_summary(session),
            "next_action": "collect_final_answers",
        }

    # ─── 6. Submit final test answers ────────────────────────────────────────
    def submit_final_test(self, session_id: str, answers: dict[str, str]) -> dict:
        """Evaluate final test → generate report → complete."""
        session = get_session(session_id)
        questions = getattr(session, "_pending_quiz", [])

        correct = 0
        for idx, q in enumerate(questions):
            user_ans = answers.get(str(idx), "").upper().strip()
            right = q.get("correct", "").upper().strip()
            is_correct = user_ans == right
            if is_correct:
                correct += 1
            attempt = QuizAttempt(
                question=q["question"],
                options=q.get("options", {}),
                correct_answer=right,
                user_answer=user_ans,
                is_correct=is_correct,
                explanation=q.get("explanation", ""),
                subtopic=q.get("subtopic", session.topic),
            )
            session.final_test_questions.append(attempt)
            session.record_quiz_answer(attempt)

        session.final_score = correct / len(questions) if questions else 0.0
        session.advance_phase(LearningPhase.COMPLETE)

        # Generate final report
        reporter = ReportAgent()
        report = reporter.generate_report(session)

        return {
            "phase": session.phase,
            "content": report,
            "questions": None,
            "session": _session_summary(session),
            "next_action": "show_report",
            "final_score": round(session.final_score * 100, 1),
        }

    # ─── 7. Reset ────────────────────────────────────────────────────────────
    def reset(self, session_id: str) -> dict:
        clear_session(session_id)
        return {"phase": LearningPhase.IDLE, "content": "Session reset. Start a new topic!", "questions": None, "session": {}, "next_action": "start_topic"}


# Singleton
_orchestrator: LearningOrchestrator | None = None

def get_orchestrator() -> LearningOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LearningOrchestrator()
    return _orchestrator
