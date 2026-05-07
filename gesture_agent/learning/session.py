from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

from gesture_agent.core.models import Intent, IntentCandidate, IntentResolution, QuestionStructure, SessionResult

from .llm_intent import LLMIntentResolver
from .question_parser import QuestionParser

FOLLOW_UP_RE = re.compile(r"(它|这个|那个|上述|刚才|继续|那|再|还有|前面|上一|区别|相比|为什么|怎么用|如何|适用|应用)")
DESIGN_EVALUATION_FOLLOW_UP_RE = re.compile(
    r"(这个方案|那个方案|刚才.*方案|上述方案|这个案例|刚才.*案例|微变|主要风险|风险|问题诊断|怎么改|修改|优化|建议|规范术语)"
)


@dataclass
class PendingClarification:
    original_query: str
    candidates: list[IntentCandidate] = field(default_factory=list)
    collected_details: list[str] = field(default_factory=list)
    attempts: int = 0
    last_question: str = ""
    memory_context: str = ""

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
    def __init__(self, parser: QuestionParser, intent_resolver: Optional[LLMIntentResolver] = None) -> None:
        self.parser = parser
        self.intent_resolver = intent_resolver
        self.pending: Optional[PendingClarification] = None
        self.turns: list[ConversationTurn] = []

    def receive(self, user_text: str, image_paths: Optional[list[str]] = None) -> SessionResult:
        image_paths = image_paths or []
        user_query = user_text.strip()
        if not user_query:
            return SessionResult(status="clarify", message="请先输入一个问题。")

        if self.pending:
            self.pending.collected_details.append(user_query)
            self.pending.attempts += 1
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
        self.turns = self.turns[-6:]

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
        if resolution.needs_clarification or resolution.intent is None:
            return SessionResult(
                status="clarify",
                message=resolution.clarification_question or "还不能确定 intent，请补充具体对象、场景或希望的输出形式。",
                user_query=user_query,
                resolved_query=query,
                memory_context=memory_context,
                resolution=resolution,
            )

        structure = self.parser.parse(query, image_paths=image_paths, forced_intent=resolution.intent)
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
        if FOLLOW_UP_RE.search(user_query) or self._carried_intent(user_query):
            return f"{user_query}\n对话记忆：\n{memory_context}"
        return user_query

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
