"""
agents/report_agent.py

Generates a comprehensive final learning report after the full test:
  - Overall score + grade
  - Per-subtopic mastery breakdown
  - Learning journey summary (how many teach/quiz rounds)
  - Strengths and persistent weaknesses
  - Specific study recommendations for exam prep
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from core.llm_factory import get_llm
from core.learning_session import LearningSession


REPORT_SYSTEM_PROMPT = """You are NeuroLearn's Assessment Director — you generate comprehensive, honest, and actionable final learning reports.

## REPORT MISSION
Give the student a complete picture of their performance and a clear path forward.

## FINAL REPORT FORMAT

---
# 📋 Final Learning Report
## Topic: {topic}

---

### 🏆 Final Score
**{score}% — {grade}**
[One sentence honest assessment of this score]

---

### 📊 Subtopic Mastery Breakdown
[Table or list — each subtopic with score and status emoji]
| Subtopic | Score | Status |
|---|---|---|
| [name] | X% | ✅ Mastered / ⚠️ Developing / ❌ Needs Work |

---

### 🛤️ Your Learning Journey
- Lessons completed: {teach_rounds}
- Quiz rounds taken: {quiz_rounds}
- Total questions attempted: {total_attempts}
- Overall improvement: [comment on progress across rounds]

---

### 💪 Your Strengths
[2-3 areas where the student performed well — be specific about what they understand]

### 🔧 Areas to Strengthen
[Specific subtopics still weak — with concrete advice on HOW to study each one]

---

### 📚 Exam Preparation Recommendations

**For the next 24 hours:**
[3-4 specific, actionable study tasks — not generic "study more"]

**High-yield topics to focus on:**
[The 2-3 most exam-critical subtopics from what was covered]

**Danger zones (common exam traps):**
[Specific misconceptions or tricky areas this student showed weakness in]

---

### 🎯 Readiness Verdict
[Honest one-paragraph assessment: are they exam-ready? What's the gap if not?]
---
"""

REPORT_HUMAN = """Generate a complete final learning report for this student.

{session_summary}

Final test results:
{test_results}

Subtopic mastery data:
{mastery_data}

Full quiz history summary:
{history_summary}
"""


def _grade(score: float) -> str:
    if score >= 90: return "A+ — Outstanding"
    if score >= 80: return "A — Excellent"
    if score >= 70: return "B — Good"
    if score >= 60: return "C — Satisfactory"
    if score >= 50: return "D — Needs Improvement"
    return "F — Significant Work Needed"


class ReportAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
        self.chain = self.llm | StrOutputParser()

    def generate_report(self, session: LearningSession) -> str:
        score_pct = int((session.final_score or 0) * 100)
        grade = _grade(session.final_score or 0)

        # Build mastery data string
        mastery_lines = []
        for name, m in session.mastery.items():
            pct = int(m.score * 100)
            status = "✅ Mastered" if pct >= 70 else ("⚠️ Developing" if pct >= 50 else "❌ Needs Work")
            mastery_lines.append(f"  {name}: {pct}% — {status} ({m.correct}/{m.attempts} correct)")

        # Test results
        test_lines = []
        for i, a in enumerate(session.final_test_questions, 1):
            s = "✅" if a.is_correct else "❌"
            test_lines.append(f"Q{i} {s} [{a.subtopic}]: {a.question[:80]}... → User: {a.user_answer}, Correct: {a.correct_answer}")

        # History
        total_attempts = len(session.quiz_attempts)
        correct_all = sum(1 for a in session.quiz_attempts if a.is_correct)

        # Fill system prompt placeholders
        prompt = REPORT_SYSTEM_PROMPT\
            .replace("{topic}", session.topic)\
            .replace("{score}", str(score_pct))\
            .replace("{grade}", grade)\
            .replace("{teach_rounds}", str(session.reteach_rounds + 1))\
            .replace("{quiz_rounds}", str(session.quiz_rounds))\
            .replace("{total_attempts}", str(total_attempts))

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=REPORT_HUMAN.format(
                session_summary=session.get_context_summary(),
                test_results="\n".join(test_lines),
                mastery_data="\n".join(mastery_lines),
                history_summary=f"{total_attempts} total questions, {correct_all} correct ({int(correct_all/total_attempts*100) if total_attempts else 0}% lifetime accuracy). Weak areas identified: {', '.join(session.weak_areas) or 'none'}.",
            ))
        ]
        return self.chain.invoke(messages)
