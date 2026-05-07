from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser


def test_compare_question_detects_targets() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("单击和长按有什么区别？")

    assert structure.intent == "interaction_compare"
    assert "单击" in structure.terms
    assert "长按" in structure.terms
    assert "interaction_mechanism" in structure.layers


def test_basic_interaction_mechanism_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("什么是单击？")

    assert structure.intent == "basic_interaction_mechanism"
    assert structure.output_frame == ["核心定义", "基础属性", "状态/变化序列", "响应逻辑", "适用与不适用", "关联机制"]


def test_advanced_interaction_mechanism_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("快击如何解决单击和长按的冲突？")

    assert structure.intent == "advanced_interaction_mechanism"
    assert "要解决的问题" in structure.output_frame


def test_control_form_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("旋钮这种控件形态可以承载哪些属性？")

    assert structure.intent == "control_form"
    assert "可用属性" in structure.output_frame


def test_basic_property_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("二元属性的关键性质是什么？")

    assert structure.intent == "basic_property"
    assert "连续性/维度/感知灵敏度" in structure.output_frame


def test_multimodal_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("多模态交互中语音和手势如何分工？")

    assert structure.intent == "multimodal_interaction"
    assert "模态组成" in structure.output_frame


def test_voice_interaction_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("语音交互需要怎样设计反馈闭环？")

    assert structure.intent == "voice_interaction"
    assert "反馈闭环" in structure.output_frame


def test_podcast_content_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("帮我写一期关于长按的播客脚本")

    assert structure.intent == "podcast_content"
    assert "示例口播" in structure.output_frame


def test_background_knowledge_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("交互的本质和操控力视角是什么？")

    assert structure.intent == "background_knowledge"
    assert "核心观点" in structure.output_frame


def test_case_question_detects_case_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("分析一下手机图标长按进入编辑模式这个案例")

    assert structure.intent == "case_analysis"
    assert "text" in structure.case_modality
    assert "案例理解" in structure.focus
    assert "控件形态" in structure.output_frame


def test_resolve_intent_requires_clarification_for_vague_query() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    resolution = parser.resolve_intent("这个怎么用？")

    assert resolution.needs_clarification is True
    assert resolution.intent is None or resolution.confidence < 0.65


def test_resolve_intent_is_ready_for_clear_query() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    resolution = parser.resolve_intent("什么是单击？")

    assert resolution.needs_clarification is False
    assert resolution.intent == "basic_interaction_mechanism"
