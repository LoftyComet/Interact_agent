from __future__ import annotations

from types import SimpleNamespace

from gesture_agent.providers.deepseek import DeepSeekClient


def test_deepseek_uses_current_thinking_payload(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    client = DeepSeekClient(api_key="test")

    assert client.chat([{"role": "user", "content": "test"}], enable_thinking=False) == "OK"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_omits_thinking_payload_when_unspecified(monkeypatch) -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    DeepSeekClient(api_key="test").chat([{"role": "user", "content": "test"}])

    assert captured["extra_body"] is None
