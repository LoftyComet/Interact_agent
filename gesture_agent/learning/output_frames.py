from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from gesture_agent.core.models import Intent, IntentOutputFrames, QuestionSubtype


DEFAULT_OUTPUT_FRAMES_FILENAME = "output_frames.json"
OUTPUT_FRAME_MODES = {"merge", "replace"}

# open_ended 没有静态框架（由大模型动态生成），不在配置中要求。
INTENTS_WITHOUT_STATIC_FRAME = {"open_ended"}

# 输出框架不再内置默认值，全部以 data/output_frames.json 为准。
DEFAULT_OUTPUT_FRAMES: dict[Intent, list[str]] = {}
DEFAULT_SUBTYPE_FRAMES: dict[QuestionSubtype, list[str]] = {}


def load_output_frames(
    data_dir: Union[str, Path] = "data",
    output_frames_path: Optional[Union[str, Path]] = None,
) -> IntentOutputFrames:
    path = Path(output_frames_path) if output_frames_path else Path(data_dir) / DEFAULT_OUTPUT_FRAMES_FILENAME
    defaults = IntentOutputFrames(
        frames=_copy_frames(DEFAULT_OUTPUT_FRAMES),
        subtype_frames=_copy_subtype_frames(DEFAULT_SUBTYPE_FRAMES),
        source="defaults",
    )
    if not path.exists():
        return defaults

    raw = json.loads(path.read_text(encoding="utf-8"))
    return output_frames_from_config(raw, source=str(path), defaults=defaults)


def output_frames_from_config(
    raw: Any,
    *,
    source: str,
    defaults: Optional[IntentOutputFrames] = None,
) -> IntentOutputFrames:
    if not isinstance(raw, dict):
        raise ValueError(f"Output frames config must be a JSON object: {source}")

    defaults = defaults or IntentOutputFrames(
        frames=_copy_frames(DEFAULT_OUTPUT_FRAMES),
        subtype_frames=_copy_subtype_frames(DEFAULT_SUBTYPE_FRAMES),
        source="defaults",
    )
    mode = str(raw.get("mode", "merge")).strip().lower()
    if mode not in OUTPUT_FRAME_MODES:
        raise ValueError(f"Unsupported output frames mode `{mode}`. Use one of: {sorted(OUTPUT_FRAME_MODES)}")

    frames_raw = raw.get("frames")
    if frames_raw is None:
        frames_raw = {key: value for key, value in raw.items() if key in _valid_intents()}
    if not isinstance(frames_raw, dict):
        raise ValueError("Output frames config `frames` must be an object.")

    custom_frames = _normalize_frames(frames_raw)
    subtype_frames_raw = raw.get("subtype_frames", {})
    if not isinstance(subtype_frames_raw, dict):
        raise ValueError("Output frames config `subtype_frames` must be an object.")
    custom_subtype_frames = _normalize_subtype_frames(subtype_frames_raw)
    if mode == "replace":
        required = _valid_intents() - INTENTS_WITHOUT_STATIC_FRAME
        missing = sorted(required - set(custom_frames))
        if missing:
            raise ValueError(f"Output frames replace mode is missing intents: {', '.join(missing)}")
        return IntentOutputFrames(
            frames=custom_frames,
            subtype_frames=custom_subtype_frames,
            source=source,
        )

    merged = _copy_frames(defaults.frames)
    for intent, frame in custom_frames.items():
        merged[intent] = frame
    merged_subtypes = _copy_subtype_frames(defaults.subtype_frames)
    merged_subtypes.update(custom_subtype_frames)
    return IntentOutputFrames(
        frames=merged,
        subtype_frames=merged_subtypes,
        source=f"{defaults.source}+{source}",
    )


def _normalize_frames(frames_raw: dict[str, Any]) -> dict[Intent, list[str]]:
    frames: dict[Intent, list[str]] = {}
    valid_intents = _valid_intents()
    for intent, frame in frames_raw.items():
        if intent not in valid_intents:
            raise ValueError(f"Unknown output frame intent `{intent}`.")
        if not isinstance(frame, list):
            raise ValueError(f"Output frame for `{intent}` must be an array.")
        normalized = [str(item).strip() for item in frame if str(item).strip()]
        if not normalized:
            raise ValueError(f"Output frame for `{intent}` must contain at least one item.")
        frames[intent] = normalized  # type: ignore[assignment]
    return frames


def _copy_frames(frames: dict[Intent, list[str]]) -> dict[Intent, list[str]]:
    return {intent: list(frame) for intent, frame in frames.items()}


def _normalize_subtype_frames(frames_raw: dict[str, Any]) -> dict[QuestionSubtype, list[str]]:
    valid_subtypes = _valid_subtypes()
    frames: dict[QuestionSubtype, list[str]] = {}
    for subtype, frame in frames_raw.items():
        if subtype not in valid_subtypes:
            raise ValueError(f"Unknown output frame subtype `{subtype}`.")
        if not isinstance(frame, list):
            raise ValueError(f"Output frame for subtype `{subtype}` must be an array.")
        normalized = [str(item).strip() for item in frame if str(item).strip()]
        if not normalized:
            raise ValueError(f"Output frame for subtype `{subtype}` must contain at least one item.")
        frames[subtype] = normalized  # type: ignore[assignment]
    return frames


def _copy_subtype_frames(
    frames: dict[QuestionSubtype, list[str]],
) -> dict[QuestionSubtype, list[str]]:
    return {subtype: list(frame) for subtype, frame in frames.items()}


def _valid_intents() -> set[str]:
    return set(Intent.__args__)  # type: ignore[attr-defined]


def _valid_subtypes() -> set[str]:
    return set(QuestionSubtype.__args__)  # type: ignore[attr-defined]
