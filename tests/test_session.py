from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.core.models import IntentResolution
from gesture_agent.learning import ConversationSession, QuestionParser


def test_ambiguous_question_requests_clarification() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    result = session.receive("这个怎么用？")

    assert result.status == "clarify"
    assert "无法确定" in result.message or "多种理解" in result.message
    assert session.pending is not None


def test_clarification_detail_resolves_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    first = session.receive("这个怎么用？")
    second = session.receive("我想问单击这个基础交互机制")

    assert first.status == "clarify"
    assert second.status == "ready"
    assert second.structure is not None
    assert second.structure.intent == "basic_interaction_mechanism"
    assert session.pending is None


def test_compare_without_targets_keeps_clarifying() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    first = session.receive("帮我对比一下")
    second = session.receive("还没想好具体对象")

    assert first.status == "clarify"
    assert "对比" in first.message
    assert second.status == "clarify"
    assert session.pending is not None


def test_follow_up_uses_previous_turn_memory() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    first = session.receive("什么是单击？")
    assert first.status == "ready"
    assert first.structure is not None
    session.record_turn(
        user_query=first.user_query,
        resolved_query=first.resolved_query,
        structure=first.structure,
        answer="单击是在短时间内完成按下和抬起的基础交互机制。",
    )

    second = session.receive("那它和长按有什么区别？")

    assert second.status == "ready"
    assert second.structure is not None
    assert second.structure.intent == "interaction_compare"
    assert "单击" in second.structure.terms
    assert "长按" in second.structure.terms
    assert "什么是单击" in second.memory_context
    assert "对话记忆" in second.resolved_query


def test_session_can_use_llm_intent_resolver() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    resolver = StubIntentResolver()
    session = ConversationSession(parser, intent_resolver=resolver)  # type: ignore[arg-type]

    result = session.receive("这个是什么意思？")

    assert result.status == "ready"
    assert result.structure is not None
    assert result.structure.intent == "background_knowledge"
    assert resolver.calls == ["这个是什么意思？"]


class StubIntentResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, query: str, *, image_paths=None, memory_context: str = "") -> IntentResolution:
        self.calls.append(query)
        return IntentResolution(
            intent="background_knowledge",
            confidence=0.92,
            needs_clarification=False,
            clarification_question="",
        )
