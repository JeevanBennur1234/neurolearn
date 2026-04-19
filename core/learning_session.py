"""
core/learning_session.py

Tracks the full adaptive learning lifecycle per user session:
  IDLE → TEACHING → MINI_QUIZ → REVIEWING_MISTAKES → RETEACHING → READY → FULL_TEST → COMPLETE

Each session stores:
  - topic, subtopics covered, mastery scores per subtopic
  - concept lesson content (so reteach knows what was already said)
  - quiz results with per-question analysis
  - weak areas identified after each quiz round
  - full history for context injection
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time, uuid


class LearningPhase(str, Enum):
    IDLE            = "idle"            # No session active
    TEACHING        = "teaching"        # Concept lesson in progress
    MINI_QUIZ       = "mini_quiz"       # Practice questions active
    REVIEWING       = "reviewing"       # Showing quiz mistakes
    RETEACHING      = "reteaching"      # Re-explaining weak areas
    READY           = "ready"           # User said they feel ready
    FULL_TEST       = "full_test"       # Final formal MCQ test
    COMPLETE        = "complete"        # Test done, report generated


@dataclass
class QuizAttempt:
    question: str
    options: dict[str, str]          # {"A": "...", "B": "...", ...}
    correct_answer: str              # "A"
    user_answer: str                 # "C"
    is_correct: bool
    explanation: str                 # Why the correct answer is right
    subtopic: str                    # Which subtopic this covers
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubtopicMastery:
    name: str
    attempts: int = 0
    correct: int = 0
    teach_rounds: int = 0            # How many times this was (re)taught

    @property
    def score(self) -> float:
        return (self.correct / self.attempts) if self.attempts > 0 else 0.0

    @property
    def is_weak(self) -> bool:
        return self.attempts > 0 and self.score < 0.7


@dataclass
class LearningSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    phase: LearningPhase = LearningPhase.IDLE
    subtopics: list[str] = field(default_factory=list)
    mastery: dict[str, SubtopicMastery] = field(default_factory=dict)

    # Content history
    lesson_content: list[str] = field(default_factory=list)    # All teach/reteach content
    quiz_attempts: list[QuizAttempt] = field(default_factory=list)
    current_quiz_batch: list[QuizAttempt] = field(default_factory=list)

    # Weak area tracking
    weak_areas: list[str] = field(default_factory=list)
    reteach_rounds: int = 0
    quiz_rounds: int = 0

    # Final test
    final_test_questions: list[QuizAttempt] = field(default_factory=list)
    final_score: Optional[float] = None

    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self):
        self.last_active = time.time()

    def add_lesson(self, content: str):
        self.lesson_content.append(content)

    def record_quiz_answer(self, attempt: QuizAttempt):
        self.quiz_attempts.append(attempt)
        self.current_quiz_batch.append(attempt)

        if attempt.subtopic not in self.mastery:
            self.mastery[attempt.subtopic] = SubtopicMastery(name=attempt.subtopic)
        m = self.mastery[attempt.subtopic]
        m.attempts += 1
        if attempt.is_correct:
            m.correct += 1

    def compute_weak_areas(self) -> list[str]:
        self.weak_areas = [
            name for name, m in self.mastery.items() if m.is_weak
        ]
        return self.weak_areas

    def get_overall_score(self) -> float:
        total = sum(m.attempts for m in self.mastery.values())
        correct = sum(m.correct for m in self.mastery.values())
        return (correct / total) if total > 0 else 0.0

    def get_context_summary(self) -> str:
        """Compact context for prompt injection."""
        lines = [
            f"Topic: {self.topic}",
            f"Phase: {self.phase}",
            f"Subtopics covered: {', '.join(self.subtopics) or 'none yet'}",
            f"Quiz rounds: {self.quiz_rounds}",
            f"Overall score: {self.get_overall_score()*100:.0f}%",
        ]
        if self.weak_areas:
            lines.append(f"Weak areas: {', '.join(self.weak_areas)}")
        if self.lesson_content:
            lines.append(f"Last lesson summary (first 400 chars): {self.lesson_content[-1][:400]}")
        return "\n".join(lines)

    def start_new_quiz_batch(self):
        self.current_quiz_batch = []
        self.quiz_rounds += 1

    def advance_phase(self, to: LearningPhase):
        self.phase = to
        self.touch()


# ── In-memory session store ────────────────────────────────────────────────────
_sessions: dict[str, LearningSession] = {}

def get_session(session_id: str) -> LearningSession:
    if session_id not in _sessions:
        _sessions[session_id] = LearningSession(session_id=session_id)
    return _sessions[session_id]

def clear_session(session_id: str):
    _sessions.pop(session_id, None)

def list_sessions() -> list[str]:
    return list(_sessions.keys())
