import json

import pytest

from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser
from gesture_agent.learning.output_frames import load_output_frames


REPLACE_FRAMES = {
    "basic_interaction_mechanism": ["核心定义", "基础属性", "响应逻辑", "交互特性", "典型案例", "适用与不适用", "关联机制"],
    "control_form": ["控件定义", "图", "可用属性", "典型案例"],
    "basic_property": ["核心含义", "图", "相关案例", "基本性质"],
    "multimodal_interaction": ["模态组成", "交互逻辑", "融合/切换逻辑", "适用场景", "案例或启发"],
    "voice_interaction": ["声音交互类型", "识别/触发逻辑", "适用场景"],
    "interaction_compare": ["对比对象", "共同基础", "核心差异", "选择建议"],
    "background_knowledge": ["背景回答"],
    "case_analysis": ["案例描述", "控件形态", "基础属性", "交互机制", "优劣势"],
    "design_evaluation": ["方案复述", "结构拆解", "问题诊断", "修改建议", "需要补充的信息"],
    "mechanism_identification": ["操作描述复述", "候选交互机制", "判断逻辑", "下一步追问"],
    "function_interaction_breakdown": ["功能概述", "情况与对应交互", "共性与差异", "小结"],
    "mechanism_parameter_compare": ["参数维度", "资料依据", "各参数取舍", "选择建议与探索提示"],
    "interaction_optimization": ["现状复述", "问题诊断", "需要澄清的信息", "优化建议"],
    "evaluation_methodology": ["资料边界说明", "评估单元", "评估维度与检查点", "评估方法建议"],
    "design_suggestion": ["需求理解", "可参考的概念与机制", "设计建议（仅供参考）", "需要进一步澄清的信息"],
    "retrieval_instruction": ["检索结果"],
}


def test_output_frames_replace_mode_applies_custom_frame(tmp_path) -> None:
    config = tmp_path / "output_frames.json"
    frames = dict(REPLACE_FRAMES)
    frames["design_evaluation"] = ["专家复述", "专家诊断", "专家建议"]
    config.write_text(
        json.dumps({"mode": "replace", "frames": frames}, ensure_ascii=False),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load("data")
    output_frames = load_output_frames("data", output_frames_path=config)
    parser = QuestionParser(kb, output_frames=output_frames)
    structure = parser.parse("请评估这个设计方案：用户长按音量旋钮后拖动来调节音量。")

    assert structure.output_frame == ["专家复述", "专家诊断", "专家建议"]
    assert parser.output_frames.frames["interaction_compare"] == ["对比对象", "共同基础", "核心差异", "选择建议"]


def test_output_frames_support_subtype_overrides(tmp_path) -> None:
    config = tmp_path / "output_frames.json"
    config.write_text(
        json.dumps({
            "mode": "merge",
            "subtype_frames": {"control_form_compare": ["形态对比", "选型建议"]},
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    output_frames = load_output_frames("data", output_frames_path=config)

    assert output_frames.frame_for("interaction_compare", "control_form_compare") == ["形态对比", "选型建议"]


def test_basic_mechanism_frame_uses_corpus_term_interaction_characteristics() -> None:
    output_frames = load_output_frames("data")

    frame = output_frames.frame_for("basic_interaction_mechanism")

    assert "交互特性" in frame
    assert "收益与代价" not in frame


def test_output_frames_config_rejects_unknown_intent(tmp_path) -> None:
    config = tmp_path / "output_frames.json"
    config.write_text(
        """
{
  "mode": "merge",
  "frames": {
    "unknown_intent": ["标题"]
  }
}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown output frame intent"):
        load_output_frames("data", output_frames_path=config)


def test_output_frames_replace_mode_requires_all_intents(tmp_path) -> None:
    config = tmp_path / "output_frames.json"
    config.write_text(
        """
{
  "mode": "replace",
  "frames": {
    "design_evaluation": ["方案复述"]
  }
}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing intents"):
        load_output_frames("data", output_frames_path=config)
