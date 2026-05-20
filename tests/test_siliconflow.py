from gesture_agent.providers.siliconflow import SiliconFlowClient


def test_siliconflow_chat_does_not_send_enable_thinking_by_default(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    client = SiliconFlowClient(api_key="test-key")
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"

    assert captured["extra_body"] is None


def test_siliconflow_chat_can_send_enable_thinking_when_configured(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    client = SiliconFlowClient(api_key="test-key")
    client.chat([{"role": "user", "content": "hi"}], enable_thinking=False)

    assert captured["extra_body"] == {"enable_thinking": False}
