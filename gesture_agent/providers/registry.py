"""Provider registry — maps a provider id to a chat-client factory.

Adding a provider = add one entry to `PROVIDERS`. The frontend gets the list via
`/api/providers` and sends back the chosen `id`; the backend resolves it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from gesture_agent.providers.anthropic_client import AnthropicClient
from gesture_agent.providers.base import ProviderError
from gesture_agent.providers.siliconflow import SiliconFlowClient
from gesture_agent.settings.env import get_env


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    api_key_env: str
    build: Callable[..., Any]

    def is_configured(self) -> bool:
        return bool(get_env(self.api_key_env))


def _build_siliconflow(model=None, base_url=None, timeout=None, use_vision_model=False):
    return SiliconFlowClient.from_env(
        model=model, base_url=base_url, timeout=timeout, use_vision_model=use_vision_model
    )


def _build_kimi(model=None, base_url=None, timeout=None, use_vision_model=False):
    # base_url is provider-specific; ignore the SiliconFlow base_url passed by callers.
    return AnthropicClient.from_env(model=model, timeout=timeout, use_vision_model=use_vision_model)


PROVIDERS: dict[str, ProviderSpec] = {
    "siliconflow": ProviderSpec(
        id="siliconflow",
        label="DeepSeek V3.2",
        api_key_env="SILICONFLOW_API_KEY",
        build=_build_siliconflow,
    ),
    "kimi": ProviderSpec(
        id="kimi",
        label="Kimi",
        api_key_env="KIMI_API_KEY",
        build=_build_kimi,
    ),
}

DEFAULT_PROVIDER = "siliconflow"


def list_providers() -> list[dict[str, Any]]:
    """Public-facing provider list for the frontend dropdown."""
    return [
        {"id": spec.id, "label": spec.label, "configured": spec.is_configured()}
        for spec in PROVIDERS.values()
    ]


def resolve_provider_id(provider_id: Optional[str]) -> str:
    if provider_id and provider_id in PROVIDERS:
        return provider_id
    return DEFAULT_PROVIDER


def build_chat_client(
    provider_id: Optional[str] = None,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
    use_vision_model: bool = False,
) -> Any:
    spec = PROVIDERS.get(resolve_provider_id(provider_id))
    if spec is None:  # pragma: no cover - resolve_provider_id guarantees a valid id
        raise ProviderError(f"Unknown provider: {provider_id}")
    return spec.build(
        model=model, base_url=base_url, timeout=timeout, use_vision_model=use_vision_model
    )
