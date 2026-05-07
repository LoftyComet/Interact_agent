from __future__ import annotations

import json
from typing import Any, Optional

from gesture_agent.core.models import Intent, IntentCandidate, IntentResolution
from gesture_agent.providers import SiliconFlowClient, SiliconFlowError

from .question_parser import INTENT_LABELS, QuestionParser


class LLMIntentResolver:
    def __init__(self, parser: QuestionParser, client: SiliconFlowClient) -> None:
        self.parser = parser
        self.client = client

    def resolve(
        self,
        query: str,
        *,
        image_paths: Optional[list[str]] = None,
        memory_context: str = "",
    ) -> IntentResolution:
        rule_resolution = self.parser.resolve_intent(query, image_paths=image_paths)
        return self._resolve_with_rule(query, rule_resolution, memory_context)

    def _resolve_with_rule(
        self,
        query: str,
        rule_resolution: IntentResolution,
        memory_context: str,
    ) -> IntentResolution:
        prompt = self._build_prompt(query, rule_resolution, memory_context)

        try:
            raw = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=0.2,
                max_tokens=700,
                enable_thinking=False,
            )
            parsed = self._parse_json(raw)
        except (SiliconFlowError, ValueError, json.JSONDecodeError):
            return rule_resolution

        intent = parsed.get("intent")
        if intent not in INTENT_LABELS:
            intent = None

        confidence = _coerce_float(parsed.get("confidence"), default=0.0)
        needs_clarification = _coerce_bool(
            parsed.get("needs_clarification"),
            default=confidence < 0.65 or intent is None,
        )
        if intent is None:
            needs_clarification = True
        clarification_question = str(parsed.get("clarification_question") or rule_resolution.clarification_question)
        reason = str(parsed.get("reason") or "大模型辅助判断")

        candidates = list(rule_resolution.candidates)
        if intent:
            candidates.insert(0, IntentCandidate(intent=intent, score=confidence, reason=reason))

        return IntentResolution(
            intent=intent,
            confidence=confidence,
            candidates=_dedupe_candidates(candidates),
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            missing_info=list(rule_resolution.missing_info),
        )

    def _build_prompt(self, query: str, rule_resolution: IntentResolution, memory_context: str) -> str:
        valid_intents = "\n".join(f"- {key}: {label}" for key, label in INTENT_LABELS.items())
        candidates = [
            {"intent": item.intent, "score": item.score, "reason": item.reason}
            for item in rule_resolution.candidates
        ]
        return f"""你是手势词典 Agent 的 Intent 判定器。请根据用户当前输入、对话记忆和规则候选，判断最合适的 intent。

可选 intent：
{valid_intents}

对话记忆：
{memory_context or "无"}

用户当前输入：
{query}

规则候选：
```json
{json.dumps(candidates, ensure_ascii=False, indent=2)}
```

判定要求：
1. 如果当前输入是追问，例如“它”“这个”“那它和长按有什么区别”，必须结合对话记忆补全指代。
2. 如果上一轮是 design_evaluation，当前输入说“这个案例/这个方案/如何应用/主要风险/怎么改/微变”，通常应继续判为 design_evaluation，而不是机械判为 case_analysis。
3. 如果能明确判断 intent，输出 needs_clarification=false。
4. 如果仍缺少关键对象或场景，输出 needs_clarification=true，并给出一个简短反问。
5. 只输出 JSON，不要输出解释文本。

JSON 格式：
{{
  "intent": "某个可选 intent 或 null",
  "confidence": 0.0,
  "needs_clarification": true,
  "clarification_question": "需要反问时的问题，否则为空字符串",
  "reason": "一句话说明判断依据"
}}"""

    def _parse_json(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("No JSON object found in LLM intent response.")
        return json.loads(text[start : end + 1])


class ClarificationIntentResolver(LLMIntentResolver):
    """Only ask the LLM when local rules would otherwise ask the user to clarify."""

    def resolve(
        self,
        query: str,
        *,
        image_paths: Optional[list[str]] = None,
        memory_context: str = "",
    ) -> IntentResolution:
        rule_resolution = self.parser.resolve_intent(query, image_paths=image_paths)
        if not rule_resolution.needs_clarification and rule_resolution.intent is not None:
            return rule_resolution
        return self._resolve_with_rule(query, rule_resolution, memory_context)


def _coerce_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if value is None:
        return default
    return bool(value)


def _dedupe_candidates(candidates: list[IntentCandidate]) -> list[IntentCandidate]:
    deduped: list[IntentCandidate] = []
    seen: set[Intent] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.intent in seen:
            continue
        seen.add(candidate.intent)
        deduped.append(candidate)
    return deduped[:5]
