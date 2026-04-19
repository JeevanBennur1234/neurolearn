"""
utils/topic_analyzer.py

Extracts structured subtopics from any topic using LLM.
Used to:
  - Break a topic into quiz-targetable subtopics
  - Ensure every subtopic gets coverage in the quiz
  - Track mastery at subtopic granularity
"""
import json, re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from core.llm_factory import get_llm


ANALYZER_SYSTEM = """You extract the key subtopics from any academic topic.

Output ONLY a JSON array of strings — the subtopic names. No explanation. No markdown.
Keep each subtopic name concise (2-5 words). Return 4-7 subtopics maximum.

Example output:
["Definition and Overview", "Types and Classification", "Key Algorithms", "Applications", "Common Pitfalls"]
"""

ANALYZER_HUMAN = "Extract the key exam-relevant subtopics for: {topic}"


def extract_subtopics(topic: str) -> list[str]:
    llm = get_llm(temperature=0.2)
    chain = llm | StrOutputParser()
    messages = [
        SystemMessage(content=ANALYZER_SYSTEM),
        HumanMessage(content=ANALYZER_HUMAN.format(topic=topic))
    ]
    raw = chain.invoke(messages)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        result = json.loads(raw)
        return [str(s) for s in result] if isinstance(result, list) else [topic]
    except Exception:
        return [topic]
