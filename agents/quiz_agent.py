"""
agents/quiz_agent.py

Generates two types of assessments:
  1. Mini-quiz (3-5 Qs) — after lesson, tests understanding, feeds weak-area detection
  2. Full test (10-15 Qs) — comprehensive, exam-standard, all subtopics

All output is structured JSON for the orchestrator to parse and track per-question.
"""
import json, re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from core.llm_factory import get_llm


QUIZ_SYSTEM_PROMPT = """You are NeuroLearn's Quiz Designer — you generate high-quality MCQs that test understanding, not just recall.

## MCQ QUALITY RULES
- Each question tests ONE specific concept — no ambiguity
- All 4 options must be plausible — no throwaway wrong answers
- Distractors must target real misconceptions
- Questions should vary: definitions, applications, comparisons, "EXCEPT" types
- Never repeat the same concept twice in one quiz

## CRITICAL OUTPUT RULE
You MUST output ONLY a valid JSON array. No preamble, no explanation, no markdown fences.
The JSON must be parseable by Python's json.loads().

## JSON FORMAT
[
  {
    "question": "Full question text here?",
    "options": {
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    },
    "correct": "B",
    "explanation": "B is correct because... A is wrong because... C is wrong because... D is wrong because...",
    "subtopic": "The specific subtopic this question tests",
    "difficulty": "medium"
  }
]
"""

FULL_TEST_SYSTEM_PROMPT = """You are NeuroLearn's Exam Paper Setter — creating a comprehensive formal test.

## TEST DESIGN RULES
- 10-15 questions covering ALL subtopics proportionally
- Mix: 30% easy recall, 50% medium application, 20% hard analysis
- Include at least 2 "EXCEPT/NOT" questions
- Include at least 2 comparison/contrast questions
- Options must be exam-realistic (no obviously wrong choices)
- Every question tests something different

## CRITICAL OUTPUT RULE
Output ONLY a valid JSON array. No preamble, no explanation, no markdown fences.

## JSON FORMAT (same as mini-quiz format — include all fields)
"""

QUIZ_HUMAN = """Generate a {count}-question mini-quiz on: {topic}

Focus on these subtopics (distribute evenly): {subtopics}

{context_block}

Student context: {session_context}

Output ONLY the JSON array. Nothing else."""

FULL_TEST_HUMAN = """Generate a comprehensive {count}-question final test on: {topic}

ALL subtopics to cover: {subtopics}
Known weak areas (include extra questions here): {weak_areas}

{context_block}

Output ONLY the JSON array. Nothing else."""


def _parse_quiz_json(raw: str) -> list[dict]:
    """Robustly extract JSON array from LLM output."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    # Find the array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)


class QuizAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.5)
        self.chain = self.llm | StrOutputParser()

    def generate_mini_quiz(
        self,
        topic: str,
        subtopics: list[str],
        session_context: str = "",
        rag_context: str = "",
        count: int = 5,
    ) -> list[dict]:
        context_block = f"\n📚 REFERENCE:\n{rag_context[:800]}\n" if rag_context.strip() else ""
        messages = [
            SystemMessage(content=QUIZ_SYSTEM_PROMPT),
            HumanMessage(content=QUIZ_HUMAN.format(
                count=count,
                topic=topic,
                subtopics=", ".join(subtopics) if subtopics else topic,
                context_block=context_block,
                session_context=session_context,
            ))
        ]
        raw = self.chain.invoke(messages)
        return _parse_quiz_json(raw)

    def generate_full_test(
        self,
        topic: str,
        subtopics: list[str],
        weak_areas: list[str],
        rag_context: str = "",
        count: int = 10,
    ) -> list[dict]:
        context_block = f"\n📚 REFERENCE:\n{rag_context[:800]}\n" if rag_context.strip() else ""
        messages = [
            SystemMessage(content=FULL_TEST_SYSTEM_PROMPT + QUIZ_SYSTEM_PROMPT),
            HumanMessage(content=FULL_TEST_HUMAN.format(
                count=count,
                topic=topic,
                subtopics=", ".join(subtopics),
                weak_areas=", ".join(weak_areas) if weak_areas else "none",
                context_block=context_block,
            ))
        ]
        raw = self.chain.invoke(messages)
        return _parse_quiz_json(raw)
