from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from gesture_agent.providers.base import ProviderError
from gesture_agent.settings.env import get_env, load_env_file


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
VISION_MODEL_ENV = "SILICONFLOW_VISION_MODEL"


class SiliconFlowError(ProviderError):
    pass


@dataclass
class SiliconFlowClient:
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
    ) -> "SiliconFlowClient":
        load_env_file()
        api_key = get_env("SILICONFLOW_API_KEY")
        if not api_key:
            raise SiliconFlowError("Missing SILICONFLOW_API_KEY. Set it in `.env` or your shell environment.")
        configured_model = get_env("SILICONFLOW_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        if use_vision_model:
            configured_model = get_env(VISION_MODEL_ENV, configured_model) or configured_model
        return cls(
            api_key=api_key,
            model=model or configured_model,
            base_url=(base_url or get_env("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
            timeout=timeout or int(get_env("SILICONFLOW_TIMEOUT", "60") or "60"),
            max_retries=int(get_env("SILICONFLOW_MAX_RETRIES", "0") or "0"),
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
            raise SiliconFlowError("Missing openai package. Run `uv sync` or install `openai`.") from exc

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        extra_body: dict[str, Any] = {}
        if enable_thinking is not None:
            extra_body["enable_thinking"] = enable_thinking

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
            raise SiliconFlowError(
                f"SiliconFlow request timed out after {self.timeout}s. "
                "Increase SILICONFLOW_TIMEOUT or use a faster/non-reasoning model."
            ) from exc
        except APIConnectionError as exc:
            raise SiliconFlowError(
                f"SiliconFlow connection error. base_url={self.base_url}. "
                "Check DNS/network/VPN/proxy with: curl -I https://api.siliconflow.cn/v1/models"
            ) from exc
        except APIStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise SiliconFlowError(f"SiliconFlow HTTP {exc.status_code}: {detail}") from exc
        except APIError as exc:
            raise SiliconFlowError(f"SiliconFlow API error: {exc}") from exc
        except Exception as exc:
            raise SiliconFlowError(f"SiliconFlow request failed: {exc}") from exc

        if not response.choices or response.choices[0].message.content is None:
            raise SiliconFlowError(f"Unexpected SiliconFlow response: {response}")
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
        extra_body: dict[str, Any] = {}
        if enable_thinking is not None:
            extra_body["enable_thinking"] = enable_thinking

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
            raise SiliconFlowError("Missing openai package. Run `uv sync` or install `openai`.") from exc
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _wrap_error(self, exc: Exception) -> SiliconFlowError:
        try:
            from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError
        except ImportError:
            return SiliconFlowError(f"SiliconFlow request failed: {exc}")

        if isinstance(exc, APITimeoutError):
            return SiliconFlowError(
                f"SiliconFlow request timed out after {self.timeout}s. "
                "Increase SILICONFLOW_TIMEOUT/--timeout, use --stream, or use a faster model."
            )
        if isinstance(exc, APIConnectionError):
            return SiliconFlowError(
                f"SiliconFlow connection error. base_url={self.base_url}. "
                "Check DNS/network/VPN/proxy with: curl -I https://api.siliconflow.cn/v1/models"
            )
        if isinstance(exc, APIStatusError):
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            return SiliconFlowError(f"SiliconFlow HTTP {exc.status_code}: {detail}")
        if isinstance(exc, APIError):
            return SiliconFlowError(f"SiliconFlow API error: {exc}")
        return SiliconFlowError(f"SiliconFlow request failed: {exc}")
