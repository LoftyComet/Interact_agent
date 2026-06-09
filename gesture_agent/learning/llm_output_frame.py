from __future__ import annotations

import json
import logging
from typing import Any, Optional

from gesture_agent.core.models import Intent, IntentOutputFrames
from gesture_agent.providers import SiliconFlowClient, SiliconFlowError

logger = logging.getLogger(__name__)

MIN_FRAME_ITEMS = 3
MAX_FRAME_ITEMS = 8

OPEN_ENDED_FALLBACK_FRAME = ["问题理解", "核心要点", "相关线索", "建议或下一步"]


class LLMOutputFrameResolver:
    def __init__(
        self,
        client: SiliconFlowClient,
        output_frames: IntentOutputFrames,
        *,
        mode: str = "auto",
        confidence_threshold: float = 0.80,
    ) -> None:
        self.client = client
        self.output_frames = output_frames
        self.mode = mode
        self.confidence_threshold = confidence_threshold

    def resolve(
        self,
        query: str,
        intent: Intent,
        confidence: float,
        memory_context: str = "",
    ) -> Optional[list[str]]:
        is_open_ended = intent == "open_ended"
        if not is_open_ended:
            if self.mode == "false":
                return None
            if self.mode == "auto" and confidence >= self.confidence_threshold:
                return None

        default_frame = self.output_frames.frame_for(intent)
        prompt = self._build_prompt(query, intent, confidence, default_frame, memory_context)

        try:
            raw = self.client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.8,
                max_tokens=500,
                enable_thinking=None,
            )
            parsed = self._parse_json(raw)
        except SiliconFlowError as exc:
            logger.warning("LLM output frame API call failed, using static frame: %s", exc)
            return list(OPEN_ENDED_FALLBACK_FRAME) if is_open_ended else None
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("LLM output frame response parse failed, using static frame: %s", exc)
            return list(OPEN_ENDED_FALLBACK_FRAME) if is_open_ended else None

        frame = parsed.get("output_frame")
        if not isinstance(frame, list):
            logger.warning("LLM output frame response missing valid output_frame list")
            return list(OPEN_ENDED_FALLBACK_FRAME) if is_open_ended else None

        frame = [str(item).strip() for item in frame if str(item).strip()]
        if len(frame) < MIN_FRAME_ITEMS or len(frame) > MAX_FRAME_ITEMS:
            logger.warning("LLM output frame has %d items (expected %d-%d), using static frame", len(frame), MIN_FRAME_ITEMS, MAX_FRAME_ITEMS)
            return list(OPEN_ENDED_FALLBACK_FRAME) if is_open_ended else None

        return frame

    def _build_prompt(
        self,
        query: str,
        intent: Intent,
        confidence: float,
        current_frame: list[str],
        memory_context: str,
    ) -> str:
        all_frames = "\n".join(
            f"- {key}: {', '.join(frame)}"
            for key, frame in self.output_frames.frames.items()
        )

        if intent == "open_ended":
            return f"""你是手势词典 Agent 的输出框架生成器。当前用户问题不属于词典内置的固定意图（开放问题），需要根据问题本身定义合适的章节结构。

参考：现有内置 intent 的输出框架风格（仅供风格参考，不要求复用）：
{all_frames}

用户问题：{query}
对话记忆：{memory_context or "无"}

请按以下步骤思考：
1. 用户的核心诉求是什么？这是一个解释类、操作类、对比类、还是创作类问题？
2. 该问题最自然的回答需要哪几个步骤或维度？
3. 输出 4-7 个章节标题，覆盖核心诉求；可以借鉴现有框架的章节命名风格，但不必拘泥于词典术语。

只输出 JSON，格式：
{{"reasoning": "一段简短推理", "output_frame": ["章节1", "章节2", ...]}}"""

        current_frame_str = "、".join(current_frame)
        return f"""你是手势词典 Agent 的输出框架生成器。用户的问题可能不完全匹配现有的固定输出框架，你需要根据问题特点生成更合适的章节结构。

现有 intent 输出框架参考：
{all_frames}

用户问题：{query}
判定 intent：{intent}（置信度：{confidence:.2f}）
当前默认框架：{current_frame_str}
对话记忆：{memory_context or "无"}

请按以下步骤思考：
1. 用户真正想了解什么？核心诉求是什么？
2. 当前的 {intent} 默认框架是否完全适用？哪些章节多余？哪些缺失？
3. 是否需要从其他 intent 框架借鉴章节？
4. 最终输出 4-7 个章节标题，确保覆盖用户核心诉求，同时尽量复用已有框架中的章节名称。

只输出 JSON，格式：
{{"reasoning": "一段简短推理", "output_frame": ["章节1", "章节2", ...]}}"""

    def _parse_json(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found in LLM output frame response.")
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
        return obj
