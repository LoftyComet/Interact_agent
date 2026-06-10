from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

from gesture_agent.core.models import Intent, IntentCandidate, IntentResolution, QuestionStructure, SessionResult

from .clarify_options import get_clarify_spec
from .llm_intent import LLMIntentResolver
from .llm_output_frame import OPEN_ENDED_FALLBACK_FRAME, LLMOutputFrameResolver
from .question_parser import QuestionParser

# 显式指代词：出现即视为延续上一话题。
ANAPHORA_RE = re.compile(r"(它|这个|那个|上述|刚才|前面|上一|继续)")
# 泛化提示词：追问里常见，但在全新问题里同样常见，单独出现不足以判定追问。
# 只有当新问题与最近几轮共享词典术语时，才把它们当作追问（轻量话题相似度门控）。
GENERIC_CUE_RE = re.compile(r"(那|再|还有|区别|相比|为什么|怎么用|如何|适用|应用)")
# 保留向后兼容：上面两个模式的并集。
FOLLOW_UP_RE = re.compile(r"(它|这个|那个|上述|刚才|继续|那|再|还有|前面|上一|区别|相比|为什么|怎么用|如何|适用|应用)")
DESIGN_EVALUATION_FOLLOW_UP_RE = re.compile(
    r"(这个方案|那个方案|刚才.*方案|上述方案|这个案例|刚才.*案例|微变|主要风险|风险|问题诊断|怎么改|修改|优化|建议|规范术语)"
)

MAX_HISTORY_TURNS = 6
MAX_CLARIFICATION_ATTEMPTS = 3
# 话题相似度判断回看的轮数。
RECENT_TOPIC_TURNS = 4


@dataclass
class PendingClarification:
    original_query: str
    candidates: list[IntentCandidate] = field(default_factory=list)
    collected_details: list[str] = field(default_factory=list)
    attempts: int = 0
    last_question: str = ""
    memory_context: str = ""
    original_intent: Optional[Intent] = None
    original_terms: list[str] = field(default_factory=list)

    def combined_query(self) -> str:
        if not self.collected_details:
            return self.original_query
        details = "；".join(self.collected_details)
        return f"{self.original_query}\n补充信息：{details}"


@dataclass
class ConversationTurn:
    user_query: str
    resolved_query: str
    structure: QuestionStructure
    answer_summary: str = ""


class ConversationSession:
    def __init__(
        self,
        parser: QuestionParser,
        intent_resolver: Optional[LLMIntentResolver] = None,
        output_frame_resolver: Optional[LLMOutputFrameResolver] = None,
    ) -> None:
        self.parser = parser
        self.intent_resolver = intent_resolver
        self.output_frame_resolver = output_frame_resolver
        self.pending: Optional[PendingClarification] = None
        self.turns: list[ConversationTurn] = []

    def receive(self, user_text: str, image_paths: Optional[list[str]] = None) -> SessionResult:
        image_paths = image_paths or []
        user_query = user_text.strip()
        if not user_query:
            return SessionResult(status="clarify", message="请先输入一个问题。")

        if self.pending:
            if self._is_topic_switch_during_clarification(user_query):
                self.pending = None
            else:
                self.pending.collected_details.append(user_query)
                self.pending.attempts += 1

                if self.pending.attempts >= MAX_CLARIFICATION_ATTEMPTS:
                    self.pending = None
                    return SessionResult(
                        status="clarify",
                        message="已多次尝试仍无法确定意图，请重新描述你的问题。",
                        user_query=user_query,
                    )

                combined_query = self.pending.combined_query()
                result = self._try_resolve(
                    combined_query,
                    image_paths,
                    user_query=combined_query,
                    memory_context=self.pending.memory_context,
                )
                if result.status == "ready":
                    self.pending = None
                    return result

                self.pending.candidates = result.resolution.candidates if result.resolution else []
                self.pending.last_question = result.message
                return result

        memory_context = self.memory_summary()
        contextual_query = self._contextualize_query(user_query, memory_context)
        carried_intent = self._carried_intent(user_query)
        result = self._try_resolve(
            contextual_query,
            image_paths,
            user_query=user_query,
            memory_context=memory_context,
            forced_intent=carried_intent,
        )
        if result.status == "clarify":
            resolution = result.resolution
            self.pending = PendingClarification(
                original_query=contextual_query,
                candidates=resolution.candidates if resolution else [],
                last_question=result.message,
                memory_context=memory_context,
                original_intent=resolution.intent if resolution else None,
                original_terms=self.parser.kb.find_terms(user_query),
            )
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
        answer: str = "",
    ) -> None:
        self.turns.append(
            ConversationTurn(
                user_query=user_query,
                resolved_query=resolved_query,
                structure=structure,
                answer_summary=_summarize_answer(answer),
            )
        )
        self.turns = self.turns[-MAX_HISTORY_TURNS:]

    def memory_summary(self, max_turns: int = 4) -> str:
        if not self.turns:
            return ""
        lines: list[str] = []
        for index, turn in enumerate(self.turns[-max_turns:], start=1):
            terms = "、".join(turn.structure.terms) if turn.structure.terms else "未命中术语"
            lines.append(
                f"{index}. 用户问：{turn.user_query}；intent={turn.structure.intent}；术语={terms}；"
                f"输出框架={'/'.join(turn.structure.output_frame)}"
            )
            if turn.answer_summary:
                lines.append(f"   上次回答摘要：{turn.answer_summary}")
        return "\n".join(lines)

    def _try_resolve(
        self,
        query: str,
        image_paths: list[str],
        *,
        user_query: str,
        memory_context: str,
        forced_intent: Optional[Intent] = None,
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
            resolution = self.intent_resolver.resolve(query, image_paths=image_paths, memory_context=memory_context)
        else:
            resolution = self.parser.resolve_intent(query, image_paths=image_paths)
        # 部分意图在作答前需要先让用户选定子类型（如语音交互的「含义识别类 / 声学控制类」）。
        # 这类意图一旦被判定为 top intent，就用选项式追问代替通用澄清：
        # - 用户已点明子类型 -> 视为意图明确，直接进入作答（忽略 needs_clarification）；
        # - 未点明 -> 返回带选项的追问，由前端渲染成可点击按钮。
        clarify_spec = get_clarify_spec(resolution.intent)
        if clarify_spec is not None:
            if clarify_spec.matched_subtype(user_query) is None:
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
            return SessionResult(
                status="clarify",
                message=resolution.clarification_question or "还不能确定 intent，请补充具体对象、场景或希望的输出形式。",
                user_query=user_query,
                resolved_query=query,
                memory_context=memory_context,
                resolution=resolution,
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

    def _contextualize_query(self, user_query: str, memory_context: str) -> str:
        if not memory_context:
            return user_query
        if self._is_follow_up(user_query):
            return f"{user_query}\n对话记忆：\n{memory_context}"
        return user_query

    def _is_follow_up(self, user_query: str) -> bool:
        """判断当前输入是否延续上一话题。

        分层判定，避免无关话题切换被误判为追问：
        1. 显式指代词（它/这个/上述…）出现即视为追问；
        2. design_evaluation 等需要延续的意图维持原有行为；
        3. 泛化提示词（为什么/如何/区别…）只有在新问题与最近几轮
           共享词典术语时才算追问（轻量话题相似度门控）。
        """
        if ANAPHORA_RE.search(user_query) or self._carried_intent(user_query):
            return True
        if GENERIC_CUE_RE.search(user_query) and self._shares_recent_topic(user_query):
            return True
        return False

    def _shares_recent_topic(self, user_query: str) -> bool:
        """新问题是否与最近几轮命中相同的词典术语。"""
        current_terms = set(self.parser.kb.find_terms(user_query))
        if not current_terms:
            return False
        recent_terms: set[str] = set()
        for turn in self.turns[-RECENT_TOPIC_TURNS:]:
            recent_terms.update(turn.structure.terms)
        return bool(current_terms & recent_terms)

    def _is_topic_switch_during_clarification(self, user_query: str) -> bool:
        """澄清状态下，判断用户是否换了一个无关的新问题（而非回答澄清）。

        保守判定，宁可继续粘合也不要误伤合法澄清：仅当
        1. 原问题没有确定意图（intent 为 None，即真正模糊的提问，
           而不是“意图已知、仅缺对象”的提问，如“帮我对比一下”）；
        2. 新输入自带词典术语；
        3. 这些术语与原问题术语无重合；
        4. 新输入不含指代词（它/这个/上述…，含则仍是延续）
        同时满足时，才判为话题切换。
        """
        if self.pending is None or self.pending.original_intent is not None:
            return False
        if ANAPHORA_RE.search(user_query):
            return False
        current_terms = set(self.parser.kb.find_terms(user_query))
        if not current_terms:
            return False
        return not (current_terms & set(self.pending.original_terms))

    def _carried_intent(self, user_query: str) -> Optional[Intent]:
        if not self.turns:
            return None
        last_intent = self.turns[-1].structure.intent
        if last_intent == "design_evaluation" and DESIGN_EVALUATION_FOLLOW_UP_RE.search(user_query):
            return "design_evaluation"
        return None


def _summarize_answer(answer: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", answer).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."
