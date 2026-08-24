from __future__ import annotations

import re
from typing import Optional

from gesture_agent.core.models import (
    ConversationTurn,
    Intent,
    IntentCandidate,
    IntentResolution,
    PendingClarification,
    QuestionStructure,
    SessionResult,
    TopicRelation,
)

from .clarify_options import get_clarify_spec
from .llm_intent import LLMIntentResolver
from .llm_output_frame import OPEN_ENDED_FALLBACK_FRAME, LLMOutputFrameResolver
from .question_parser import INTENT_LABELS, QuestionParser
from .turn_classifier import LLM_FALLBACK_THRESHOLD, TurnClassifier

MAX_HISTORY_TURNS = 6
MAX_CLARIFICATION_ATTEMPTS = 3


class ConversationSession:
    def __init__(
        self,
        parser: QuestionParser,
        intent_resolver: Optional[LLMIntentResolver] = None,
        output_frame_resolver: Optional[LLMOutputFrameResolver] = None,
        turn_classifier: Optional[TurnClassifier] = None,
        dynamic_clarify_question: bool = True,
        intent_candidate_options: bool = True,
    ) -> None:
        self.parser = parser
        self.intent_resolver = intent_resolver
        self.output_frame_resolver = output_frame_resolver
        self.classifier = turn_classifier or TurnClassifier(parser)
        self.pending: Optional[PendingClarification] = None
        self.turns: list[ConversationTurn] = []
        self.dynamic_clarify_question = dynamic_clarify_question
        self.intent_candidate_options = intent_candidate_options

    def receive(self, user_text: str, image_paths: Optional[list[str]] = None,
                forced_intent: Optional[Intent] = None) -> SessionResult:
        image_paths = image_paths or []
        user_query = user_text.strip()
        if not user_query:
            return SessionResult(status="clarify", message="请先输入一个问题。")

        decision = self.classifier.classify(user_query, self.turns, self.pending)

        # —— 澄清态分发：回答澄清 vs 切换话题 ——
        if self.pending is not None:
            if decision.relation == TopicRelation.CLARIFY_REPLY:
                return self._continue_clarification(user_query, image_paths)
            # NEW_TOPIC：放弃澄清，按全新问题重新判定话题关系。
            self.pending = None
            decision = self.classifier.classify(user_query, self.turns, None)

        # —— 正常态：维度 A 决定是否注入记忆，维度 B 由解析层产出 ——
        memory_context = self.memory_summary()
        contextual_query = self._contextualize_query(
            user_query, memory_context, decision.relation, decision.confidence
        )
        result = self._try_resolve(
            contextual_query,
            image_paths,
            user_query=user_query,
            memory_context=memory_context,
            forced_intent=forced_intent or decision.carried_intent,
        )
        if result.status == "clarify":
            resolution = result.resolution
            self.pending = PendingClarification(
                original_query=contextual_query,
                candidates=resolution.candidates if resolution else [],
                last_question=result.message,
                question_history=[result.message],
                memory_context=memory_context,
                original_intent=resolution.intent if resolution else None,
                original_terms=self.parser.kb.find_terms(user_query),
            )
        return result

    _INTENT_SELECTION_RE = re.compile(r"【intent:(.+?)】")

    def _continue_clarification(self, user_query: str, image_paths: list[str]) -> SessionResult:
        """澄清态下，把当前输入作为对澄清的补充并尝试重新解析。"""
        assert self.pending is not None

        # 检测用户是否通过点击意图候选选项来选定 intent
        intent_match = self._INTENT_SELECTION_RE.search(user_query)
        selected_intent: Optional[Intent] = None
        if intent_match:
            raw_intent = intent_match.group(1)
            if raw_intent in INTENT_LABELS:
                selected_intent = raw_intent
            user_query = self._INTENT_SELECTION_RE.sub("", user_query).strip()

        self.pending.collected_details.append(user_query)
        self.pending.attempts += 1

        if self.pending.attempts >= MAX_CLARIFICATION_ATTEMPTS:
            self.pending = None
            return SessionResult(
                status="clarify",
                message="已多次尝试仍无法确定意图，请重新描述你的问题。",
                user_query=user_query,
            )

        clarification_history = list(self.pending.question_history)
        combined_query = self.pending.combined_query()
        result = self._try_resolve(
            combined_query,
            image_paths,
            user_query=combined_query,
            memory_context=self.pending.memory_context,
            clarification_history=clarification_history,
            forced_intent=selected_intent,
        )
        if result.status == "ready":
            self.pending = None
            return result

        self.pending.candidates = result.resolution.candidates if result.resolution else []
        self.pending.last_question = result.message
        self.pending.question_history.append(result.message)
        return result

    def reset(self) -> None:
        self.pending = None
        self.turns = []

    def record_turn(
        self,
        *,
        user_query: str,
        resolved_query: str,
        structure: QuestionStructure,
    ) -> None:
        self.turns.append(
            ConversationTurn(
                user_query=user_query,
                resolved_query=resolved_query,
                structure=structure,
            )
        )
        self.turns = self.turns[-MAX_HISTORY_TURNS:]

    def memory_summary(self, max_turns: int = 4) -> str:
        """生成记忆摘要：最近 N 轮用户问题的编号列表。"""
        if not self.turns:
            return ""
        turns_to_use = self.turns[-max_turns:]
        lines = [f"{i}. {turn.user_query}" for i, turn in enumerate(turns_to_use, start=1)]
        return "\n".join(lines)

    def _try_resolve(
        self,
        query: str,
        image_paths: list[str],
        *,
        user_query: str,
        memory_context: str,
        forced_intent: Optional[Intent] = None,
        clarification_history: Optional[list[str]] = None,
    ) -> SessionResult:
        if forced_intent:
            resolution = IntentResolution(
                intent=forced_intent,
                confidence=0.93,
                candidates=[
                    IntentCandidate(
                        intent=forced_intent,
                        score=0.93,
                        reason="根据上一轮会话意图和当前追问延续判断。",
                    )
                ],
                needs_clarification=False,
            )
        elif self.intent_resolver:
            resolution = self.intent_resolver.resolve(
                query,
                image_paths=image_paths,
                memory_context=memory_context,
                clarification_history=clarification_history if self.dynamic_clarify_question else None,
            )
        else:
            resolution = self.parser.resolve_intent(
                query,
                image_paths=image_paths,
                clarification_history=clarification_history if self.dynamic_clarify_question else None,
            )
        # 部分意图在作答前需要先让用户选定子类型（如语音交互的「含义识别类 / 声学控制类」）。
        # 这类意图一旦被判定为 top intent，就用选项式追问代替通用澄清：
        # - 用户已点明子类型 -> 视为意图明确，直接进入作答（忽略 needs_clarification）；
        # - 未点明 -> 返回带选项的追问，由前端渲染成可点击按钮。
        clarify_spec = get_clarify_spec(resolution.intent)
        if clarify_spec is not None:
            if clarify_spec.should_clarify(user_query):
                return SessionResult(
                    status="clarify",
                    message=clarify_spec.message,
                    user_query=user_query,
                    resolved_query=query,
                    memory_context=memory_context,
                    resolution=resolution,
                    options=[opt.to_dict() for opt in clarify_spec.options],
                )
        elif resolution.needs_clarification or resolution.intent is None:
            # 检查是否可以展示意图候选选项（仅在启用了 intent_candidate_options 且真正接近时）
            candidate_options: list[dict[str, str]] = []
            if (
                self.intent_candidate_options
                and resolution.candidates
                and resolution.intent is not None
                and self.parser._is_ambiguous_call(resolution.candidates)
            ):
                candidate_options = self.parser._build_intent_candidate_options(
                    user_query, resolution.candidates
                )

            message = (
                "这个问题可能有多种理解，请选择最接近的一种："
                if candidate_options
                else (
                    resolution.clarification_question
                    or "还不能确定 intent，请补充具体对象、场景或希望的输出形式。"
                )
            )

            return SessionResult(
                status="clarify",
                message=message,
                user_query=user_query,
                resolved_query=query,
                memory_context=memory_context,
                resolution=resolution,
                options=candidate_options,
            )

        output_frame_override = None
        if self.output_frame_resolver and resolution.intent:
            output_frame_override = self.output_frame_resolver.resolve(
                query, resolution.intent, resolution.confidence, memory_context
            )
        if resolution.intent == "open_ended" and not output_frame_override:
            output_frame_override = list(OPEN_ENDED_FALLBACK_FRAME)

        structure = self.parser.parse(
            query, image_paths=image_paths, forced_intent=resolution.intent,
            output_frame_override=output_frame_override,
        )
        structure.raw_query = user_query
        return SessionResult(
            status="ready",
            user_query=user_query,
            resolved_query=query,
            memory_context=memory_context,
            structure=structure,
            resolution=resolution,
        )

    def _contextualize_query(
        self, user_query: str, memory_context: str, relation: TopicRelation, confidence: float = 0.0
    ) -> str:
        """根据话题关系 + 置信度分层注入记忆。

        - confidence >= 0.9: full — 完整 4 轮记忆含摘要
        - LLM_FALLBACK_THRESHOLD <= confidence < 0.9: medium — 2 轮不含摘要
        - confidence < LLM_FALLBACK_THRESHOLD: light — 仅上一轮意图+术语
        - NEW_TOPIC / 无记忆：不注入
        """
        if memory_context and relation == TopicRelation.FOLLOW_UP:
            if confidence >= 0.9:
                mem = self.memory_summary(max_turns=4)
            elif confidence >= LLM_FALLBACK_THRESHOLD:
                mem = self.memory_summary(max_turns=2)
            else:
                mem = self.memory_summary(max_turns=1)
            return f"{user_query}\n对话记忆：\n{mem}"
        return user_query
