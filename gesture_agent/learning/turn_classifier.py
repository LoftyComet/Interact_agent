"""话题关系判定器（维度 A：输入侧）。

把「当前输入与历史对话是什么关系」这件事，从 ``ConversationSession`` 里散落的
几个布尔函数收敛成一个统一入口 ``TurnClassifier.classify``，输出三选一的
``TopicRelation`` 加置信度。它只回答话题关系，不回答「能不能作答」——后者是
维度 B（信息充分性），由意图解析层产出。

分层判定，从强信号到弱信号；每层带置信度。低置信度时（第 5/6 层）可降级到
LLM 三分类（见 ``llm_relation``）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from gesture_agent.core.models import (
    ConversationTurn,
    Intent,
    PendingClarification,
    TopicRelation,
    TurnDecision,
)
from gesture_agent.providers import SiliconFlowClient, SiliconFlowError

from .question_parser import QuestionParser

logger = logging.getLogger(__name__)

# 显式指代词：出现即视为延续上一话题。
ANAPHORA_RE = re.compile(r"(它|这个|那个|上述|刚才|前面|上一|继续)")
# 泛化提示词：追问里常见，但在全新问题里同样常见，单独出现不足以判定追问。
# 只有当新问题与最近几轮共享词典术语时，才把它们当作追问（轻量话题相似度门控）。
GENERIC_CUE_RE = re.compile(r"(那|再|还有|区别|相比|为什么|怎么用|如何|适用|应用)")
DESIGN_EVALUATION_FOLLOW_UP_RE = re.compile(
    r"(这个方案|那个方案|刚才.*方案|上述方案|这个案例|刚才.*案例|微变|主要风险|风险|问题诊断|怎么改|修改|优化|建议|规范术语)"
)

# 话题相似度判断回看的轮数。
RECENT_TOPIC_TURNS = 4
# 低于该置信度时，可考虑降级到 LLM 判定。
LLM_FALLBACK_THRESHOLD = 0.65


class TurnClassifier:
    """判定一轮输入与历史的话题关系。"""

    def __init__(
        self,
        parser: QuestionParser,
        relation_resolver: Optional["TurnRelationResolver"] = None,
    ) -> None:
        self.parser = parser
        # 可选：低置信度时的 LLM 降级判定器（第 5 步接入，平移阶段为 None）。
        self.relation_resolver = relation_resolver

    def classify(
        self,
        user_text: str,
        turns: list[ConversationTurn],
        pending: Optional[PendingClarification],
    ) -> TurnDecision:
        # —— 澄清态：先区分「回答澄清」还是「换了新话题」 ——
        if pending is not None:
            if self._is_topic_switch_during_clarification(user_text, pending):
                return TurnDecision(
                    relation=TopicRelation.NEW_TOPIC,
                    confidence=0.8,
                    reason="澄清状态下自带无重合的新术语且无指代词，判为话题切换。",
                    signals=["clarify_topic_switch"],
                )
            return TurnDecision(
                relation=TopicRelation.CLARIFY_REPLY,
                confidence=0.85,
                reason="澄清状态下的输入，按对当前澄清的回答处理。",
                signals=["clarify_reply"],
            )

        # —— 正常态：判定追问 / 新话题 ——
        carried = self._carried_intent(user_text, turns)
        if ANAPHORA_RE.search(user_text):
            return TurnDecision(
                relation=TopicRelation.FOLLOW_UP,
                confidence=0.95,
                reason="出现显式指代词，视为延续上一话题。",
                signals=["anaphora"],
                carried_intent=carried,
            )
        if carried is not None:
            return TurnDecision(
                relation=TopicRelation.FOLLOW_UP,
                confidence=0.9,
                reason="上一轮意图可延续（如 design_evaluation 的风险/优化追问）。",
                signals=["carried_intent"],
                carried_intent=carried,
            )
        if GENERIC_CUE_RE.search(user_text) and self._shares_recent_topic(user_text, turns):
            return TurnDecision(
                relation=TopicRelation.FOLLOW_UP,
                confidence=0.8,
                reason="泛化提示词且与最近几轮共享词典术语，判为追问。",
                signals=["generic_cue", "shared_topic"],
            )

        # —— 弱信号兜底：默认新话题，必要时降级到 LLM ——
        decision = TurnDecision(
            relation=TopicRelation.NEW_TOPIC,
            confidence=0.5,
            reason="无指代词、无可延续意图、无术语重合，默认新话题。",
            signals=["default"],
        )
        if self.relation_resolver is not None and decision.confidence < LLM_FALLBACK_THRESHOLD:
            llm_decision = self.relation_resolver.resolve(user_text, turns)
            if llm_decision is not None:
                return llm_decision
        return decision

    def _shares_recent_topic(self, user_text: str, turns: list[ConversationTurn]) -> bool:
        """新问题是否与最近几轮命中相同的词典术语。"""
        current_terms = set(self.parser.kb.find_terms(user_text))
        if not current_terms:
            return False
        recent_terms: set[str] = set()
        for turn in turns[-RECENT_TOPIC_TURNS:]:
            recent_terms.update(turn.structure.terms)
        return bool(current_terms & recent_terms)

    def _is_topic_switch_during_clarification(
        self, user_text: str, pending: PendingClarification
    ) -> bool:
        """澄清状态下，判断用户是否换了一个无关的新问题（而非回答澄清）。

        保守判定，宁可继续粘合也不要误伤合法澄清：仅当
        1. 原问题没有确定意图（intent 为 None，即真正模糊的提问，
           而不是“意图已知、仅缺对象”的提问，如“帮我对比一下”）；
        2. 新输入自带词典术语；
        3. 这些术语与原问题术语无重合；
        4. 新输入不含指代词（它/这个/上述…，含则仍是延续）
        同时满足时，才判为话题切换。
        """
        if pending.original_intent is not None:
            return False
        if ANAPHORA_RE.search(user_text):
            return False
        current_terms = set(self.parser.kb.find_terms(user_text))
        if not current_terms:
            return False
        return not (current_terms & set(pending.original_terms))

    def _carried_intent(self, user_text: str, turns: list[ConversationTurn]) -> Optional[Intent]:
        if not turns:
            return None
        last_intent = turns[-1].structure.intent
        if last_intent == "design_evaluation" and DESIGN_EVALUATION_FOLLOW_UP_RE.search(user_text):
            return "design_evaluation"
        return None


class TurnRelationResolver:
    """LLM 三分类降级判定的协议占位（duck-typed）。

    实现需提供 ``resolve(user_text, turns) -> Optional[TurnDecision]``；返回 None
    表示放弃判定（如 API 失败），由调用方回落到规则默认值。
    """

    def resolve(
        self, user_text: str, turns: list[ConversationTurn]
    ) -> Optional[TurnDecision]:  # pragma: no cover - 占位
        raise NotImplementedError


# LLM 输出标签 -> TopicRelation。澄清回答只在 pending 态出现，不在弱信号兜底层判定，
# 故这里只接受 follow_up / new_topic 两类。
_LLM_RELATION_MAP: dict[str, TopicRelation] = {
    "follow_up": TopicRelation.FOLLOW_UP,
    "new_topic": TopicRelation.NEW_TOPIC,
}
# 降级时只看最近几轮，prompt 尽量短。
_LLM_CONTEXT_TURNS = 2


class LLMTurnRelationResolver:
    """规则置信度过低时，用 LLM 做「追问 vs 新话题」二选一的兜底判定。"""

    def __init__(self, client: SiliconFlowClient) -> None:
        self.client = client

    def resolve(self, user_text: str, turns: list[ConversationTurn]) -> Optional[TurnDecision]:
        if not turns:
            # 无历史，无所谓追问，直接交还规则默认（新话题）。
            return None
        prompt = self._build_prompt(user_text, turns)
        try:
            raw = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=0.2,
                max_tokens=120,
                enable_thinking=None,
            )
            parsed = self._parse_json(raw)
        except SiliconFlowError as exc:
            logger.warning("LLM turn-relation call failed, fall back to rule: %s", exc)
            return None
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("LLM turn-relation parse failed, fall back to rule: %s", exc)
            return None

        relation = _LLM_RELATION_MAP.get(str(parsed.get("relation", "")).strip().lower())
        if relation is None:
            return None
        confidence = _coerce_float(parsed.get("confidence"), default=0.66)
        reason = str(parsed.get("reason") or "大模型辅助判断话题关系")
        return TurnDecision(
            relation=relation,
            confidence=confidence,
            reason=reason,
            signals=["llm_fallback"],
        )

    def _build_prompt(self, user_text: str, turns: list[ConversationTurn]) -> str:
        history_lines = []
        for index, turn in enumerate(turns[-_LLM_CONTEXT_TURNS:], start=1):
            terms = "、".join(turn.structure.terms) if turn.structure.terms else "无"
            history_lines.append(f"{index}. 用户问：{turn.user_query}（术语：{terms}）")
        history = "\n".join(history_lines) or "无"
        return f"""你在判断用户最新一句话与上文的关系，用于决定要不要把历史对话作为上下文带入。

最近对话：
{history}

用户最新输入：
{user_text}

请二选一：
- follow_up：在追问、延续上文同一话题（即使没有出现“它/这个”等指代词）。
- new_topic：开启了一个与上文无关的新问题。

只输出 JSON，不要解释：
{{"relation": "follow_up 或 new_topic", "confidence": 0.0, "reason": "一句话依据"}}"""

    def _parse_json(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found in LLM turn-relation response.")
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
        return obj


def _coerce_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
