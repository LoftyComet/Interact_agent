from __future__ import annotations

import re

from gesture_agent.core.models import DesignEvaluationStructure


CONTROL_FORM_TERMS = [
    "按钮",
    "拨钮",
    "滚轮",
    "摇杆",
    "轨迹球",
    "指点杆",
    "触控面",
    "旋钮",
    "手柄",
    "踏板",
    "手势",
    "图标",
    "菜单",
    "滑块",
]

BASIC_PROPERTY_TERMS = [
    "二元属性",
    "多级属性",
    "位置属性",
    "角度属性",
    "力属性",
    "声音属性",
    "光属性",
    "温度属性",
    "形变属性",
    "时间属性",
    "生理信号",
    "位置",
    "角度",
    "力度",
    "压力",
    "时间",
]

MECHANISM_TERMS = [
    "单击",
    "双击",
    "长按",
    "按下",
    "开关",
    "拖拽",
    "拖动",
    "甩动",
    "滑动",
    "轻扫",
    "翻动",
    "捏合",
    "旋转",
    "快击",
    "点击缓冲",
    "点拖互斥",
    "长按拖拽",
    "双按拖拽",
]

FEEDBACK_TERMS = [
    "反馈",
    "提示",
    "弹出",
    "高亮",
    "震动",
    "声音",
    "动效",
    "变化",
    "显示",
    "隐藏",
    "确认",
    "报错",
    "取消",
]


def parse_design_evaluation(
    proposal: str,
    *,
    image_paths: list[str],
    terms: list[str],
) -> DesignEvaluationStructure:
    modality = _detect_modality(proposal, image_paths)
    control_forms = _dedupe(_find_terms(proposal, CONTROL_FORM_TERMS) + _filter_terms(terms, CONTROL_FORM_TERMS))
    basic_properties = _dedupe(_find_terms(proposal, BASIC_PROPERTY_TERMS) + _filter_terms(terms, BASIC_PROPERTY_TERMS))
    mechanisms = _dedupe(_find_terms(proposal, MECHANISM_TERMS) + _filter_terms(terms, MECHANISM_TERMS))
    feedback = _find_feedback(proposal)
    product_context = _extract_context(proposal)
    user_goal = _extract_user_goal(proposal)
    missing_info = _detect_missing_info(
        proposal=proposal,
        image_paths=image_paths,
        product_context=product_context,
        user_goal=user_goal,
        control_forms=control_forms,
        mechanisms=mechanisms,
        feedback=feedback,
    )

    return DesignEvaluationStructure(
        raw_proposal=proposal,
        modality=modality,
        product_context=product_context,
        user_goal=user_goal,
        control_forms=control_forms,
        basic_properties=basic_properties,
        mechanisms=mechanisms,
        system_feedback=feedback,
        risk_points=_detect_risks(proposal, missing_info),
        missing_info=missing_info,
    )


def _detect_modality(proposal: str, image_paths: list[str]) -> list[str]:
    modality: list[str] = []
    if image_paths or any(word in proposal for word in ["图片", "图中", "截图", "界面图"]):
        modality.append("image")
    if proposal.strip():
        modality.append("text")
    return modality or ["text"]


def _find_terms(text: str, vocabulary: list[str]) -> list[str]:
    return [term for term in vocabulary if term in text]


def _filter_terms(terms: list[str], vocabulary: list[str]) -> list[str]:
    return [term for term in terms if term in vocabulary]


def _find_feedback(text: str) -> list[str]:
    return _find_terms(text, FEEDBACK_TERMS)


def _extract_context(text: str) -> str:
    patterns = [
        r"(?:在|用于|面向)([^。；\n]{2,36})(?:中|里|场景|界面)",
        r"(?:产品|界面|场景)[：:是为]*([^。；\n]{2,36})",
    ]
    return _first_match(text, patterns)


def _extract_user_goal(text: str) -> str:
    patterns = [
        r"(?:用户|使用者)(?:可以|需要|想要|要|通过)([^。；\n]{2,42})",
        r"(?:目标|目的|任务)[：:是为]*([^。；\n]{2,42})",
        r"(?:来|以便)([^，,。；\n]{2,42})",
        r"(?:用来|用于)([^。；\n]{2,42})",
    ]
    return _first_match(text, patterns)


def _first_match(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" ，,：:；;。")
    return ""


def _detect_missing_info(
    *,
    proposal: str,
    image_paths: list[str],
    product_context: str,
    user_goal: str,
    control_forms: list[str],
    mechanisms: list[str],
    feedback: list[str],
) -> list[str]:
    missing: list[str] = []
    if not image_paths and len(proposal.strip()) < 24:
        missing.append("设计方案描述较短，缺少可评估的操作链路。")
    if not product_context and not image_paths:
        missing.append("缺少产品/界面/使用场景。")
    if not user_goal:
        missing.append("缺少用户目标或任务。")
    if not control_forms:
        missing.append("缺少控件形态。")
    if not mechanisms:
        missing.append("缺少明确的交互机制。")
    if not feedback:
        missing.append("缺少系统反馈或状态变化。")
    return missing


def _detect_risks(proposal: str, missing_info: list[str]) -> list[str]:
    risks: list[str] = []
    if "单击" in proposal and "长按" in proposal:
        risks.append("单击与长按共存时，需要用时间阈值区分快击和持续按压。")
    if "单击" in proposal and "双击" in proposal:
        risks.append("单击与双击共存时，单击响应可能需要点击缓冲，避免提前触发。")
    if any(word in proposal for word in ["点击", "单击"]) and any(word in proposal for word in ["拖拽", "拖动"]):
        risks.append("点击与拖拽共存时，需要定义点拖互斥或位移阈值。")
    if "滑动" in proposal and "拖拽" in proposal:
        risks.append("滑动与拖拽的动作相似，需要明确方向、距离或目标对象。")
    if missing_info:
        risks.append("方案信息不完整，模型应先标出缺失项，再给条件性建议。")
    return _dedupe(risks)


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped
