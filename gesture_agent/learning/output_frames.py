from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from gesture_agent.core.models import Intent, IntentOutputFrames


DEFAULT_OUTPUT_FRAMES_FILENAME = "output_frames.json"
OUTPUT_FRAME_MODES = {"merge", "replace"}

DEFAULT_OUTPUT_FRAMES: dict[Intent, list[str]] = {
    "basic_interaction_mechanism": ["核心定义", "基础属性", "状态/变化序列", "响应逻辑", "适用与不适用", "关联机制"],
    "advanced_interaction_mechanism": ["要解决的问题", "构成机制", "判定条件", "响应逻辑", "设计收益与代价", "关联基础机制"],
    "control_form": ["控件定义", "可用属性", "可承载的交互机制", "典型案例", "设计注意点"],
    "basic_property": ["核心含义", "连续性/维度/感知灵敏度", "相关案例", "适用与不适用", "可组合方向"],
    "multimodal_interaction": ["模态组成", "信息分工", "融合/切换逻辑", "适用场景", "风险与校准", "案例或启发"],
    "voice_interaction": ["输入内容与声学属性", "识别/触发逻辑", "反馈闭环", "适用场景", "限制与替代入口"],
    "podcast_content": ["主题定位", "听众对象", "内容大纲", "关键讲述点", "示例口播", "延伸问题"],
    "interaction_compare": ["对比对象", "共同基础", "核心差异", "适用边界", "选择建议"],
    "background_knowledge": ["背景问题", "核心观点", "词典中的位置", "为什么重要", "与后续知识的关系"],
    "case_analysis": ["案例描述", "控件形态", "基础属性", "交互机制", "响应逻辑", "设计判断", "追问"],
    "design_evaluation": ["方案复述", "结构拆解", "问题诊断", "修改建议", "规范术语版本", "需要补充的信息"],
    "dictionary_methodology": ["提问切入点", "词典立场", "分类逻辑", "与常见做法的差异", "学习者收益"],
}


def load_output_frames(
    data_dir: Union[str, Path] = "data",
    output_frames_path: Optional[Union[str, Path]] = None,
) -> IntentOutputFrames:
    path = Path(output_frames_path) if output_frames_path else Path(data_dir) / DEFAULT_OUTPUT_FRAMES_FILENAME
    defaults = IntentOutputFrames(frames=_copy_frames(DEFAULT_OUTPUT_FRAMES), source="defaults")
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

    defaults = defaults or IntentOutputFrames(frames=_copy_frames(DEFAULT_OUTPUT_FRAMES), source="defaults")
    mode = str(raw.get("mode", "merge")).strip().lower()
    if mode not in OUTPUT_FRAME_MODES:
        raise ValueError(f"Unsupported output frames mode `{mode}`. Use one of: {sorted(OUTPUT_FRAME_MODES)}")

    frames_raw = raw.get("frames")
    if frames_raw is None:
        frames_raw = {key: value for key, value in raw.items() if key in _valid_intents()}
    if not isinstance(frames_raw, dict):
        raise ValueError("Output frames config `frames` must be an object.")

    custom_frames = _normalize_frames(frames_raw)
    if mode == "replace":
        missing = sorted(_valid_intents() - set(custom_frames))
        if missing:
            raise ValueError(f"Output frames replace mode is missing intents: {', '.join(missing)}")
        return IntentOutputFrames(frames=custom_frames, source=source)

    merged = _copy_frames(defaults.frames)
    for intent, frame in custom_frames.items():
        merged[intent] = frame
    return IntentOutputFrames(frames=merged, source=f"{defaults.source}+{source}")


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


def _valid_intents() -> set[str]:
    return set(Intent.__args__)  # type: ignore[attr-defined]
