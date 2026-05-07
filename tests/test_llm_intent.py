from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import ClarificationIntentResolver, QuestionParser


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


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        return self.response
