from gesture_agent.providers.anthropic_client import (
    AnthropicClient,
    _split_system,
    to_anthropic_messages,
)
from gesture_agent.providers.base import ProviderError
from gesture_agent.providers.registry import (
    DEFAULT_PROVIDER,
    build_chat_client,
    list_providers,
    resolve_provider_id,
)
from gesture_agent.providers.siliconflow import SiliconFlowError


def test_split_system_pulls_out_system_messages() -> None:
    messages = [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "hi"},
    ]
    system, rest = _split_system(messages)
    assert system == "you are a bot"
    assert rest == [{"role": "user", "content": "hi"}]


def test_to_anthropic_messages_drops_system_and_keeps_turns() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    converted = to_anthropic_messages(messages)
    assert converted == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_to_anthropic_messages_converts_data_url_image() -> None:
    data_url = "data:image/png;base64,QUJD"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    [msg] = to_anthropic_messages(messages)
    text_block, image_block = msg["content"]
    assert text_block == {"type": "text", "text": "look"}
    assert image_block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_to_anthropic_messages_converts_http_image_url() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
        }
    ]
    [msg] = to_anthropic_messages(messages)
    assert msg["content"][0] == {
        "type": "image",
        "source": {"type": "url", "url": "https://x/y.png"},
    }


def test_siliconflow_error_is_provider_error() -> None:
    assert issubclass(SiliconFlowError, ProviderError)


def test_resolve_provider_id_falls_back_to_default() -> None:
    assert resolve_provider_id(None) == DEFAULT_PROVIDER
    assert resolve_provider_id("nope") == DEFAULT_PROVIDER
    assert resolve_provider_id("kimi") == "kimi"


def test_list_providers_shape() -> None:
    providers = list_providers()
    ids = {p["id"] for p in providers}
    assert {"siliconflow", "kimi"} <= ids
    for p in providers:
        assert {"id", "label", "configured"} <= p.keys()
