from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from gesture_agent.providers.base import ProviderError
from gesture_agent.settings.env import get_env, load_env_file


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
VISION_MODEL_ENV = "DEEPSEEK_VISION_MODEL"
DEFAULT_EMBEDDING_MODEL = "deepseek-v4-flash"
EMBEDDING_MODEL_ENV = "DEEPSEEK_EMBEDDING_MODEL"


class DeepSeekError(ProviderError):
    pass


@dataclass
class DeepSeekClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 120
    max_retries: int = 0
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @classmethod
    def from_env(
        cls,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        use_vision_model: bool = False,
    ) -> "DeepSeekClient":
        load_env_file()
        api_key = get_env("DEEPSEEK_API_KEY")
        if not api_key:
            raise DeepSeekError("Missing DEEPSEEK_API_KEY. Set it in `.env` or your shell environment.")
        configured_model = get_env("DEEPSEEK_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        if use_vision_model:
            configured_model = get_env(VISION_MODEL_ENV, configured_model) or configured_model
        return cls(
            api_key=api_key,
            model=model or configured_model,
            base_url=(base_url or get_env("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
            timeout=timeout or int(get_env("DEEPSEEK_TIMEOUT", "60") or "60"),
            max_retries=int(get_env("DEEPSEEK_MAX_RETRIES", "0") or "0"),
            embedding_model=get_env(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1600,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        try:
            from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
        except ImportError as exc:
            raise DeepSeekError("Missing openai package. Run `uv sync` or install `openai`.") from exc

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        extra_body = _thinking_extra_body(enable_thinking)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        except APITimeoutError as exc:
            raise DeepSeekError(
                f"DeepSeek request timed out after {self.timeout}s. "
                "Increase DEEPSEEK_TIMEOUT or use a faster/non-reasoning model."
            ) from exc
        except APIConnectionError as exc:
            raise DeepSeekError(
                f"DeepSeek connection error. base_url={self.base_url}. "
                "Check DNS/network/VPN/proxy with: curl -I https://api.deepseek.com/v1/models"
            ) from exc
        except APIStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise DeepSeekError(f"DeepSeek HTTP {exc.status_code}: {detail}") from exc
        except APIError as exc:
            raise DeepSeekError(f"DeepSeek API error: {exc}") from exc
        except Exception as exc:
            raise DeepSeekError(f"DeepSeek request failed: {exc}") from exc

        if not response.choices or response.choices[0].message.content is None:
            raise DeepSeekError(f"Unexpected DeepSeek response: {response}")
        return response.choices[0].message.content

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1600,
        enable_thinking: Optional[bool] = None,
    ) -> Iterator[str]:
        client = self._openai_client()
        extra_body = _thinking_extra_body(enable_thinking)

        finish_reason: Optional[str] = None
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta.content
                if delta:
                    yield delta
            if finish_reason == "length":
                yield (
                    "\n\n[回答已截断：达到 max_tokens 上限。"
                    "请调大 --max-tokens，例如 --max-tokens 1200。]"
                )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    def embed(
        self,
        texts: list[str],
        *,
        model: Optional[str] = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """对一批文本求 embedding 向量,按输入顺序返回。

        默认使用 self.embedding_model(DEEPSEEK_EMBEDDING_MODEL,默认
        deepseek-chat)。分批发送以控制单次请求体积。
        """
        if not texts:
            return []
        client = self._openai_client()
        use_model = model or self.embedding_model
        out: list[list[float]] = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = client.embeddings.create(model=use_model, input=batch)
                out.extend(item.embedding for item in resp.data)
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        return out

    def list_models(self, limit: int = 20) -> list[str]:
        client = self._openai_client()
        try:
            response = client.models.list()
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        model_ids: list[str] = []
        for item in response.data[:limit]:
            model_id = getattr(item, "id", None)
            if model_id:
                model_ids.append(model_id)
        return model_ids

    def _openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DeepSeekError("Missing openai package. Run `uv sync` or install `openai`.") from exc
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _wrap_error(self, exc: Exception) -> DeepSeekError:
        try:
            from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError
        except ImportError:
            return DeepSeekError(f"DeepSeek request failed: {exc}")

        if isinstance(exc, APITimeoutError):
            return DeepSeekError(
                f"DeepSeek request timed out after {self.timeout}s. "
                "Increase DEEPSEEK_TIMEOUT/--timeout, use --stream, or use a faster model."
            )
        if isinstance(exc, APIConnectionError):
            return DeepSeekError(
                f"DeepSeek connection error. base_url={self.base_url}. "
                "Check DNS/network/VPN/proxy with: curl -I https://api.deepseek.com/v1/models"
            )
        if isinstance(exc, APIStatusError):
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            return DeepSeekError(f"DeepSeek HTTP {exc.status_code}: {detail}")
        if isinstance(exc, APIError):
            return DeepSeekError(f"DeepSeek API error: {exc}")
        return DeepSeekError(f"DeepSeek request failed: {exc}")


def _thinking_extra_body(enable_thinking: Optional[bool]) -> dict[str, Any]:
    if enable_thinking is None:
        return {}
    return {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}
