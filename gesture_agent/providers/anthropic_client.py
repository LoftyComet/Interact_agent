"""Anthropic-compatible chat client (used for Kimi coding plan, etc.).

Mirrors the public surface of `SiliconFlowClient` — `chat()` and `chat_stream()`
take the same OpenAI-style message dicts so the rest of the pipeline doesn't need
to know which provider is active. Internally it translates those messages into the
Anthropic Messages API shape (system prompt split out, image_url parts converted to
base64 image source blocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from gesture_agent.providers.base import ProviderError
from gesture_agent.settings.env import get_env, load_env_file


DEFAULT_BASE_URL = "https://api.moonshot.cn/anthropic"
DEFAULT_MODEL = "kimi-k2-turbo-preview"


class AnthropicError(ProviderError):
    pass


# Default max_tokens the Anthropic API *requires* (OpenAI lets it be optional).
DEFAULT_MAX_TOKENS = 1600


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Pull every `system` message out into a single system string."""
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        else:
            rest.append(msg)
    return "\n\n".join(p for p in system_parts if p), rest


def _convert_content(content: Any) -> Any:
    """Convert OpenAI message content into Anthropic content blocks.

    - Plain strings pass through unchanged.
    - `{"type": "text", ...}` parts pass through.
    - `{"type": "image_url", "image_url": {"url": ...}}` parts become Anthropic
      image blocks. data: URLs become base64 `source` blocks; http(s) URLs become
      url `source` blocks.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            blocks.append({"type": "text", "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            blocks.append(_image_block(url))
    return blocks


def _image_block(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        # data:<media_type>;base64,<data>
        try:
            header, data = url.split(",", 1)
            media_type = header[len("data:") :].split(";", 1)[0] or "image/png"
        except ValueError:
            media_type, data = "image/png", ""
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-style messages into Anthropic Messages format."""
    _, rest = _split_system(messages)
    converted: list[dict[str, Any]] = []
    for msg in rest:
        role = msg.get("role", "user")
        if role not in {"user", "assistant"}:
            role = "user"
        converted.append({"role": role, "content": _convert_content(msg.get("content", ""))})
    return converted


@dataclass
class AnthropicClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 120
    max_retries: int = 0

    @classmethod
    def from_env(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        use_vision_model: bool = False,
    ) -> "AnthropicClient":
        load_env_file()
        api_key = get_env("KIMI_API_KEY")
        if not api_key:
            raise AnthropicError("Missing KIMI_API_KEY. Set it in `.env` or your shell environment.")
        configured_model = get_env("KIMI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        if use_vision_model:
            configured_model = get_env("KIMI_VISION_MODEL", configured_model) or configured_model
        return cls(
            api_key=api_key,
            model=model or configured_model,
            base_url=(base_url or get_env("KIMI_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
            timeout=timeout or int(get_env("KIMI_TIMEOUT", "60") or "60"),
            max_retries=int(get_env("KIMI_MAX_RETRIES", "0") or "0"),
        )

    def _client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise AnthropicError("Missing anthropic package. Run `uv sync` or install `anthropic`.") from exc
        # Moonshot/Kimi's Anthropic-compatible endpoint authenticates via
        # `Authorization: Bearer <key>`. Passing `api_key=` makes the SDK send an
        # `x-api-key` header instead, which Moonshot rejects with HTTP 401. Use
        # `auth_token=` so the SDK emits the Bearer header.
        return Anthropic(
            auth_token=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        client = self._client()
        system, converted = _split_system(messages)
        try:
            response = client.messages.create(
                model=self.model,
                system=system or None,
                messages=to_anthropic_messages(messages),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        text = "".join(parts)
        if not text:
            raise AnthropicError(f"Unexpected Anthropic response: {response}")
        if getattr(response, "stop_reason", None) == "max_tokens":
            text += "\n\n[回答已截断：达到 max_tokens 上限。请调大 --max-tokens。]"
        return text

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        enable_thinking: Optional[bool] = None,
    ) -> Iterator[str]:
        client = self._client()
        system, _ = _split_system(messages)
        try:
            with client.messages.stream(
                model=self.model,
                system=system or None,
                messages=to_anthropic_messages(messages),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            ) as stream:
                for delta in stream.text_stream:
                    if delta:
                        yield delta
                final = stream.get_final_message()
            if getattr(final, "stop_reason", None) == "max_tokens":
                yield "\n\n[回答已截断：达到 max_tokens 上限。请调大 --max-tokens。]"
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    def _wrap_error(self, exc: Exception) -> AnthropicError:
        if isinstance(exc, AnthropicError):
            return exc
        try:
            from anthropic import APIConnectionError, APIError, APIStatusError, APITimeoutError
        except ImportError:
            return AnthropicError(f"Anthropic request failed: {exc}")

        if isinstance(exc, APITimeoutError):
            return AnthropicError(
                f"Anthropic request timed out after {self.timeout}s. "
                "Increase KIMI_TIMEOUT/--timeout or use --stream."
            )
        if isinstance(exc, APIConnectionError):
            return AnthropicError(
                f"Anthropic connection error. base_url={self.base_url}. "
                "Check DNS/network/VPN/proxy and that KIMI_BASE_URL points to the Anthropic-compatible endpoint."
            )
        if isinstance(exc, APIStatusError):
            detail = exc.response.text[:500] if getattr(exc, "response", None) is not None else str(exc)
            return AnthropicError(f"Anthropic HTTP {exc.status_code}: {detail}")
        if isinstance(exc, APIError):
            return AnthropicError(f"Anthropic API error: {exc}")
        return AnthropicError(f"Anthropic request failed: {exc}")
