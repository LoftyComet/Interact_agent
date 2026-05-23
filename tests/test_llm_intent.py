from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import ClarificationIntentResolver, QuestionParser
from gesture_agent.providers import SiliconFlowError


def test_clarification_resolver_skips_llm_when_rule_is_clear() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    client = FakeClient('{"intent":"background_knowledge","confidence":0.9,"needs_clarification":false}')
    resolver = ClarificationIntentResolver(parser, client)  # type: ignore[arg-type]

    resolution = resolver.resolve("什么是单击？")

    assert resolution.intent == "basic_interaction_mechanism"
    assert resolution.needs_clarification is False
    assert client.calls == 0


def test_clarification_resolver_asks_llm_when_rule_needs_clarification() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    client = FakeClient(
        '{"intent":"design_evaluation","confidence":0.88,"needs_clarification":false,'
        '"clarification_question":"","reason":"用户省略了细节，但表达是在请求评估方案"}'
    )
    resolver = ClarificationIntentResolver(parser, client)  # type: ignore[arg-type]

    resolution = resolver.resolve("帮我评估这个方案")

    assert resolution.intent == "design_evaluation"
    assert resolution.needs_clarification is False
    assert client.calls == 1


def test_resolver_falls_back_to_rules_on_api_error() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    client = ErrorClient(SiliconFlowError("API unreachable"))
    resolver = ClarificationIntentResolver(parser, client)  # type: ignore[arg-type]

    resolution = resolver.resolve("帮我评估这个方案")

    # Should fall back gracefully without raising
    assert resolution is not None


def test_resolver_falls_back_to_rules_on_malformed_json() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    client = FakeClient("这不是 JSON 内容，完全没有括号")
    resolver = ClarificationIntentResolver(parser, client)  # type: ignore[arg-type]

    resolution = resolver.resolve("帮我评估这个方案")

    assert resolution is not None


def test_resolver_handles_nested_braces_in_reason() -> None:
    # Verify _parse_json uses raw_decode, not rfind("}") which would misparse nested braces
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    client = FakeClient(
        '{"intent":"design_evaluation","confidence":0.85,"needs_clarification":false,'
        '"clarification_question":"","reason":"含有嵌套的JSON示例：{\\"key\\":\\"value\\"}"}'
    )
    resolver = ClarificationIntentResolver(parser, client)  # type: ignore[arg-type]

    resolution = resolver.resolve("帮我评估这个方案")

    assert resolution.intent == "design_evaluation"


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        return self.response


class ErrorClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def chat(self, messages, **kwargs) -> str:
        raise self.exc

