from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

from gesture_agent.core.models import Intent, IntentCandidate, IntentResolution
from gesture_agent.providers import SiliconFlowClient, SiliconFlowError

from .question_parser import CONFIDENCE_THRESHOLD, INTENT_LABELS, QuestionParser


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
                enable_thinking=None,
            )
            parsed = self._parse_json(raw)
        except SiliconFlowError as exc:
            logger.warning("LLM intent API call failed, falling back to rule-based: %s", exc)
            return rule_resolution
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("LLM intent response parse failed, falling back to rule-based: %s", exc)
            return rule_resolution

        intent = parsed.get("intent")
        if intent not in INTENT_LABELS:
            intent = None

        confidence = _coerce_float(parsed.get("confidence"), default=0.0)
        needs_clarification = _coerce_bool(
            parsed.get("needs_clarification"),
            default=confidence < CONFIDENCE_THRESHOLD or intent is None,
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
        reference_block = self._build_reference_block(query)
        return f"""你是手势词典 Agent 的 Intent 判定器。请根据用户当前输入、对话记忆和规则候选，判断最合适的 intent。

可选 intent：
{valid_intents}
{reference_block}
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
3. 如果用户问题与手势词典/交互设计的其他预定义类别都不沾边（例如闲聊、问编程语言、问天气、问与交互设计无关的常识等），返回 `intent="open_ended"`，并把 `needs_clarification` 设为 false。注意：只要问题与手势、控件、交互机制、设计评估、多模态/语音、词典背景知识等任何一项相关，就应优先匹配对应的预定义 intent，不要轻易判为 open_ended。
4. 如果能明确判断 intent，输出 needs_clarification=false。
5. 如果仍缺少关键对象或场景，输出 needs_clarification=true，并给出一个简短反问。
6. 上方“参考样例”是人工标注的优质问题及其正确 intent，请将其作为重要参考；当用户输入与某条样例高度相似时，应倾向采用该样例的 intent。
7. 区分 control_form_compare 与 interaction_compare：若对比对象是硬件/控件载体（手表 vs 手套、头部 vs 腿部控制），判 control_form_compare；若对比的是交互机制本身（拖拽 vs 滑动），判 interaction_compare。
8. 区分 mechanism_identification 与 basic_interaction_mechanism：若用户给一段操作描述、问“这属于什么交互机制/能不能生成表达式”（反推），判 mechanism_identification；若用户已点名某个已知机制要讲解，判 basic_interaction_mechanism。
9. 区分 function_interaction_breakdown 与 case_analysis：若拆解“一类功能/多种情况”的交互枚举（涉及哪些手势、各种情况下的交互逻辑系统），判 function_interaction_breakdown；若拆解单个具体案例，判 case_analysis。
10. 区分 mechanism_parameter_compare 与 interaction_compare：若对比的是同一机制的不同参数（拖拽长距离 vs 短距离），判 mechanism_parameter_compare。
11. 区分 control_form_application 与 control_form：若问的是控件与放置位置、人群（老年人/儿童）、应用场景的关系（书中通常没有），判 control_form_application；若问控件本身的定义/属性/可承载机制，判 control_form。
12. 区分 interaction_optimization 与 design_evaluation：若用户已有一个在用的交互、说它有具体问题想优化提升（误触、不顺手），判 interaction_optimization；若用户给出一个待评审的完整设计方案，判 design_evaluation。
13. 区分 evaluation_methodology 与 design_evaluation：若问的是“如何评估交互/评估维度/好坏标准”这类方法论，判 evaluation_methodology；若是评估某个具体方案，判 design_evaluation。
14. 只输出 JSON，不要输出解释文本。

JSON 格式：
{{
  "intent": "某个可选 intent 或 null",
  "confidence": 0.0,
  "needs_clarification": true,
  "clarification_question": "需要反问时的问题，否则为空字符串",
  "reason": "一句话说明判断依据"
}}"""

    def _build_reference_block(self, query: str) -> str:
        matches = self.parser.relevant_examples(query)
        if not matches:
            return ""
        lines = [
            f"- 「{match.example.question}」→ {match.example.intent}"
            f"（{INTENT_LABELS.get(match.example.intent, match.example.intent)}）"
            for match in matches
        ]
        body = "\n".join(lines)
        return f"\n参考样例（人工标注的优质问题及正确 intent）：\n{body}\n"

    def _parse_json(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found in LLM intent response.")
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
        return obj


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
