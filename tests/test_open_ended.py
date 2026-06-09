from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import (
    ClarificationIntentResolver,
    ConversationSession,
    LLMOutputFrameResolver,
    QuestionParser,
)
from gesture_agent.learning.llm_output_frame import OPEN_ENDED_FALLBACK_FRAME
from gesture_agent.learning.output_frames import load_output_frames


class ScriptedClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        if not self.responses:
            raise AssertionError("ScriptedClient ran out of canned responses")
        return self.responses.pop(0)


def _make_session(intent_response, frame_response):
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    intent_client = ScriptedClient([intent_response])
    intent_resolver = ClarificationIntentResolver(parser, intent_client)  # type: ignore[arg-type]
    frame_client = ScriptedClient([frame_response])
    frame_resolver = LLMOutputFrameResolver(
        frame_client,  # type: ignore[arg-type]
        load_output_frames("data"),
        mode="auto",
    )
    return ConversationSession(parser, intent_resolver=intent_resolver, output_frame_resolver=frame_resolver)


def test_open_ended_triggers_dynamic_frame() -> None:
    intent_json = (
        '{"intent":"open_ended","confidence":0.9,"needs_clarification":false,'
        '"clarification_question":"","reason":"问题与手势词典无关"}'
    )
    frame_json = (
        '{"reasoning":"普通问答","output_frame":["问题理解","解题思路","代码示例","注意事项"]}'
    )
    session = _make_session(intent_json, frame_json)

    result = session.receive("用 Python 写个快排")

    assert result.status == "ready"
    assert result.structure is not None
    assert result.structure.intent == "open_ended"
    assert result.structure.output_frame == ["问题理解", "解题思路", "代码示例", "注意事项"]
    assert result.structure.output_frame_source == "cot"


def test_open_ended_falls_back_when_frame_resolver_returns_invalid() -> None:
    intent_json = (
        '{"intent":"open_ended","confidence":0.85,"needs_clarification":false,'
        '"clarification_question":"","reason":"非词典问题"}'
    )
    # 故意返回不到 3 个章节，触发兜底
    frame_json = '{"output_frame":["唯一章节"]}'
    session = _make_session(intent_json, frame_json)

    result = session.receive("今天北京天气怎么样")

    assert result.status == "ready"
    assert result.structure is not None
    assert result.structure.intent == "open_ended"
    assert result.structure.output_frame == OPEN_ENDED_FALLBACK_FRAME
    assert result.structure.output_frame_source == "cot"


def test_open_ended_uses_fallback_when_no_frame_resolver() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    intent_json = (
        '{"intent":"open_ended","confidence":0.9,"needs_clarification":false,'
        '"clarification_question":"","reason":"非词典问题"}'
    )
    intent_resolver = ClarificationIntentResolver(parser, ScriptedClient([intent_json]))  # type: ignore[arg-type]
    session = ConversationSession(parser, intent_resolver=intent_resolver, output_frame_resolver=None)

    result = session.receive("帮我推荐一首歌")

    assert result.status == "ready"
    assert result.structure is not None
    assert result.structure.intent == "open_ended"
    assert result.structure.output_frame == OPEN_ENDED_FALLBACK_FRAME


def test_predefined_intent_keeps_static_frame() -> None:
    # 即使 intent_resolver 存在，规则已经能确定 intent 时，ClarificationIntentResolver 不调用 LLM；
    # 输出框架也应当保持 static
    intent_json = '{"intent":"open_ended","confidence":0.9,"needs_clarification":false}'  # 不应被使用
    frame_json = '{"output_frame":["不该被使用"]}'  # 不应被使用
    session = _make_session(intent_json, frame_json)

    result = session.receive("长按和双击有什么区别？")

    assert result.status == "ready"
    assert result.structure is not None
    assert result.structure.intent == "interaction_compare"
    assert result.structure.output_frame_source == "static"
    assert result.structure.output_frame == ["对比对象", "共同基础", "核心差异", "选择建议"]
