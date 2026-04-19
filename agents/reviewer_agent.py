"""
agents/reviewer_agent.py

After each quiz batch, analyzes ALL mistakes:
  - Explains WHY each answer was wrong
  - Identifies the root misconception
  - Groups mistakes by subtopic
  - Generates a diagnostic "what to study next" recommendation
  - Gives the student an honest performance summary
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from core.llm_factory import get_llm
from core.learning_session import QuizAttempt


REVIEW_SYSTEM_PROMPT = """You are NeuroLearn's Learning Coach — you analyze quiz mistakes and give honest, constructive, personalized feedback.

## YOUR MISSION
Turn mistakes into learning moments. Be honest about performance but encouraging about improvement.
Never be harsh. Never be vague. Every piece of feedback must be ACTIONABLE.

## REVIEW FORMAT

---
## 📊 Quiz Results

**Score: {score}% ({correct}/{total} correct)**

{score_message}

---

### ✅ What You Got Right
[Briefly praise correct answers — mention the concept they demonstrated understanding of]

### ❌ Mistakes — Let's Fix Them

[For each wrong answer:]

**Q: "{question_text}"**
- You answered: **(X) {user_answer_text}**
- Correct answer: **(Y) {correct_answer_text}**
- 🔍 Why you were wrong: [Specific reason — what the wrong option actually represents, why it's tempting]
- 💡 The key insight: [The core principle that makes the correct answer right — in one clear sentence]
- 🧠 How to remember: [A quick mental hook, pattern, or rule]

---

### 📌 Weak Areas Identified
[List subtopics where performance was poor — be specific about which part of that subtopic]

### 🎯 What Happens Next
[Tell them: if score ≥ 70%: encouragement + option to continue; if score < 70%: specific re-teaching areas]
---

## TONE RULES
- Be honest about poor performance — don't sugarcoat a 40% score
- Be genuinely encouraging about improvement
- Use "you" language — personal, direct
- Never say "great job" for a low score — say "let's fix this"
"""

REVIEW_HUMAN = """Analyze these quiz results and generate detailed feedback.

Topic: {topic}
Quiz Round: {round_num}

RESULTS:
{results_text}

Session context:
{session_context}

Generate the full review following the format exactly."""


def _format_results(attempts: list[QuizAttempt]) -> str:
    lines = []
    for i, a in enumerate(attempts, 1):
        status = "✅ CORRECT" if a.is_correct else "❌ WRONG"
        lines.append(
            f"Q{i} [{status}] (subtopic: {a.subtopic})\n"
            f"  Question: {a.question}\n"
            f"  Options: {a.options}\n"
            f"  Correct: {a.correct_answer} | User answered: {a.user_answer}\n"
            f"  Explanation: {a.explanation}\n"
        )
    return "\n".join(lines)


class ReviewerAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
        self.chain = self.llm | StrOutputParser()

    def review(
        self,
        topic: str,
        attempts: list[QuizAttempt],
        round_num: int,
        session_context: str = "",
    ) -> str:
        if not attempts:
            return "No quiz attempts to review."

        correct = sum(1 for a in attempts if a.is_correct)
        total = len(attempts)
        score_pct = int(correct / total * 100)

        results_text = _format_results(attempts)

        # Inject score into system prompt for tone calibration
        prompt = REVIEW_SYSTEM_PROMPT.replace("{score}", str(score_pct))
        prompt = prompt.replace("{correct}", str(correct))
        prompt = prompt.replace("{total}", str(total))

        if score_pct >= 80:
            score_message = "**Excellent work!** You've demonstrated strong understanding."
        elif score_pct >= 60:
            score_message = "**Decent attempt.** A few areas need more attention before you're test-ready."
        else:
            score_message = "**Needs work.** Don't worry — this is exactly what practice is for. Let's fix the gaps."

        prompt = prompt.replace("{score_message}", score_message)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=REVIEW_HUMAN.format(
                topic=topic,
                round_num=round_num,
                results_text=results_text,
                session_context=session_context,
            ))
        ]
        return self.chain.invoke(messages)
