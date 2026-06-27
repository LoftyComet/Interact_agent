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


def test_topic_switch_during_clarification_drops_pending() -> None:
    """澄清状态下换无关新问题：丢弃旧 pending，不再当作补充信息粘合。"""
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    first = session.receive("这个怎么用？")
    assert first.status == "clarify"
    assert session.pending is not None
    assert session.pending.original_intent is None

    # 自带“语音”术语、与原模糊问题无术语重合、无指代词 → 判为话题切换。
    second = session.receive("语音交互为什么需要唤醒词？")

    # 旧问题不应被粘进新输入。
    assert "这个怎么用" not in (second.resolved_query or "")
    assert "补充信息" not in (second.resolved_query or "")


def test_clarification_answer_with_shared_intent_still_combines() -> None:
    """原问题意图已知（仅缺对象）时，后续输入仍按澄清粘合，不误判为切换。"""
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    first = session.receive("帮我对比一下")
    assert first.status == "clarify"
    assert session.pending is not None
    # “帮我对比一下”意图明确为 interaction_compare，只是缺对比对象。
    assert session.pending.original_intent == "interaction_compare"

    second = session.receive("单击和长按")

    assert second.status == "ready"
    assert second.structure is not None
    assert second.structure.intent == "interaction_compare"
    assert session.pending is None


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


def test_design_evaluation_follow_up_keeps_evaluation_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    first = session.receive("请评估这个设计方案：在音乐播放器界面，用户长按音量旋钮后拖动来调节音量，松手后系统高亮确认。")
    assert first.status == "ready"
    assert first.structure is not None
    assert first.structure.intent == "design_evaluation"
    session.record_turn(
        user_query=first.user_query,
        resolved_query=first.resolved_query,
        structure=first.structure,
        answer="建议将长按拖拽改为更直接的旋转拖拽，并使用微变响应提供连续反馈。",
    )

    second = session.receive("如何应用微变")

    assert second.status == "ready"
    assert second.structure is not None
    assert second.structure.intent == "design_evaluation"
    assert second.structure.design_evaluation is not None
    assert second.structure.design_evaluation.product_context == "音乐播放器"
    assert second.structure.design_evaluation.user_goal == "调节音量"
    assert "对话记忆" in second.resolved_query


def test_design_evaluation_case_word_follow_up_does_not_fall_back_to_case_analysis() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    session = ConversationSession(parser)

    first = session.receive("请评估这个设计方案：在音乐播放器界面，用户长按音量旋钮后拖动来调节音量，松手后系统高亮确认。")
    assert first.status == "ready"
    assert first.structure is not None
    session.record_turn(
        user_query=first.user_query,
        resolved_query=first.resolved_query,
        structure=first.structure,
        answer="评估完成。",
    )

    second = session.receive("就是刚才那个案例如何应用微变")

    assert second.status == "ready"
    assert second.structure is not None
    assert second.structure.intent == "design_evaluation"
    assert session.pending is None


def test_unrelated_topic_switch_does_not_attach_memory() -> None:
    """切换到无关话题时，即便带泛化词（为什么/如何），也不应拼接上一话题记忆。"""
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

    # 新问题命中“语音”术语，与上一轮“单击”无交集 → 不算追问。
    second = session.receive("语音交互为什么需要唤醒词？")

    # 无论意图是否需要澄清，resolved_query 都不应带上一话题记忆。
    assert "对话记忆" not in second.resolved_query


def test_generic_cue_with_shared_term_is_follow_up() -> None:
    """泛化词 + 与上一话题共享术语 → 仍判为追问并拼接记忆。"""
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

    second = session.receive("单击为什么这么常用？")

    # 与上一话题共享“单击”术语 → resolved_query 应带上记忆。
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

    def resolve(self, query: str, *, image_paths=None, memory_context: str = "", clarification_history=None) -> IntentResolution:
        self.calls.append(query)
        return IntentResolution(
            intent="background_knowledge",
            confidence=0.92,
            needs_clarification=False,
            clarification_question="",
        )
