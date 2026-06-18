from __future__ import annotations
import json
import os
import re
import time

from openai import OpenAI
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("FPT_API_KEY")
        if not api_key:
            raise ValueError("FPT_API_KEY environment variable not set")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://mkp-api.fptcloud.com",
        )
    return _client

MODEL = "gpt-oss-120b"

def _chat(system: str, user: str) -> tuple[str, int, int]:
    """Returns (content, total_tokens, latency_ms)."""
    client = _get_client()
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    content = resp.choices[0].message.content or ""
    total_tokens = resp.usage.total_tokens if resp.usage else 0
    return content, total_tokens, latency_ms

def _extract_json(text: str) -> dict:
    """Extract first JSON object from text (handles markdown code blocks)."""
    # Try to find JSON in ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in response: {text[:200]}")

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int]:
    context_str = "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}" for chunk in example.context
    )
    reflection_block = ""
    if reflection_memory:
        reflection_block = "\n\nPrevious reflection notes:\n" + "\n".join(
            f"- {r}" for r in reflection_memory
        )

    user_msg = (
        f"Context:\n{context_str}"
        f"{reflection_block}\n\n"
        f"Question: {example.question}"
    )

    content, tokens, latency = _chat(ACTOR_SYSTEM, user_msg)

    # Parse "Final Answer: ..." from response
    match = re.search(r"Final Answer[:\s]+(.+)", content, re.IGNORECASE)
    answer = match.group(1).strip() if match else content.strip()
    return answer, tokens, latency

def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )
    content, tokens, latency = _chat(EVALUATOR_SYSTEM, user_msg)
    try:
        data = _extract_json(content)
        return JudgeResult(**data), tokens, latency
    except Exception:
        # Fallback: simple string match
        from .utils import normalize_answer
        score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
        reason = "Exact match fallback (JSON parse failed)."
        return JudgeResult(score=score, reason=reason), tokens, latency

def reflector(
    example: QAExample, attempt_id: int, judge: JudgeResult
) -> tuple[ReflectionEntry, int, int]:
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Wrong predicted answer: (see evaluator reason)\n"
        f"Evaluator reason: {judge.reason}"
    )
    content, tokens, latency = _chat(REFLECTOR_SYSTEM, user_msg)
    try:
        data = _extract_json(content)
        return (
            ReflectionEntry(attempt_id=attempt_id, **data),
            tokens,
            latency,
        )
    except Exception:
        return (
            ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Re-read all context passages carefully.",
                next_strategy="Identify each hop explicitly before giving the final answer.",
            ),
            tokens,
            latency,
        )
