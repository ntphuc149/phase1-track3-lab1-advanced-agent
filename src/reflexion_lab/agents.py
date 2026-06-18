from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Literal
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

def _use_mock() -> bool:
    return os.environ.get("MOCK_MODE", "0") == "1"

def _load_mock_runtime():
    from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
    def _actor(example, attempt_id, agent_type, reflection_memory):
        return actor_answer(example, attempt_id, agent_type, reflection_memory), 320 + attempt_id * 65, 160 + attempt_id * 40
    def _evaluator(example, answer):
        return evaluator(example, answer), 120, 90
    def _reflector(example, attempt_id, judge):
        return reflector(example, attempt_id, judge), 150, 80
    return _actor, _evaluator, _reflector, FAILURE_MODE_BY_QID

def _load_llm_runtime():
    from .llm_runtime import actor_answer, evaluator, reflector
    # No FAILURE_MODE_BY_QID for real runs — determined from judge result
    return actor_answer, evaluator, reflector, {}

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        if _use_mock():
            actor_fn, evaluator_fn, reflector_fn, failure_map = _load_mock_runtime()
        else:
            actor_fn, evaluator_fn, reflector_fn, failure_map = _load_llm_runtime()

        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        for attempt_id in range(1, self.max_attempts + 1):
            answer, token_estimate, latency_ms = actor_fn(
                example, attempt_id, self.agent_type, reflection_memory
            )
            judge, eval_tokens, eval_latency = evaluator_fn(example, answer)
            token_estimate += eval_tokens
            latency_ms += eval_latency

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, ref_tokens, ref_latency = reflector_fn(example, attempt_id, judge)
                reflections.append(reflection)
                trace.reflection = reflection
                trace.token_estimate += ref_tokens
                trace.latency_ms += ref_latency
                reflection_memory.append(
                    f"Attempt {attempt_id} failed. Lesson: {reflection.lesson} "
                    f"Next strategy: {reflection.next_strategy}"
                )
            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = (
            "none"
            if final_score == 1
            else failure_map.get(example.qid, "wrong_final_answer")
        )
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
