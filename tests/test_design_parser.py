from __future__ import annotations

from gesture_agent.evaluation.design_parser import parse_design_evaluation


def _parse(proposal: str, image_paths: list[str] | None = None, terms: list[str] | None = None):
    return parse_design_evaluation(proposal, image_paths=image_paths or [], terms=terms or [])


def test_detects_control_forms_and_mechanisms() -> None:
    result = _parse("用户长按旋钮后拖拽来调节音量，系统显示进度条反馈。")
    assert "旋钮" in result.control_forms
    assert "长按" in result.mechanisms
    assert "拖拽" in result.mechanisms
    assert "显示" in result.system_feedback


def test_detects_risk_for_click_and_drag_coexistence() -> None:
    result = _parse("用户可以单击按钮，也可以拖拽按钮移动位置。")
    assert any("点拖互斥" in r for r in result.risk_points)


def test_detects_risk_for_single_and_double_click() -> None:
    result = _parse("短按按钮单击触发，连续双击进入编辑模式，系统高亮确认。")
    assert any("点击缓冲" in r for r in result.risk_points)


def test_short_proposal_reports_missing_info() -> None:
    result = _parse("滑动")
    assert any("描述较短" in item for item in result.missing_info)


def test_empty_proposal_reports_missing_info() -> None:
    result = _parse("")
    assert result.missing_info


def test_image_only_suppresses_length_check() -> None:
    # When an image is provided, the "description too short" rule shouldn't fire for empty text
    result = _parse("", image_paths=["screen.png"])
    missing_subjects = [m for m in result.missing_info if "描述较短" in m]
    assert not missing_subjects


def test_modality_text_only() -> None:
    result = _parse("用户拖拽滑块来调节亮度。")
    assert result.modality == ["text"]


def test_modality_image_and_text() -> None:
    result = _parse("如图所示，用户拖拽旋钮。", image_paths=["design.png"])
    assert "image" in result.modality
    assert "text" in result.modality


def test_extracts_product_context() -> None:
    result = _parse("在音乐播放器界面中，用户长按旋钮调节音量。")
    assert result.product_context


def test_risk_incomplete_info_flag() -> None:
    # A proposal missing all key components should add the "信息不完整" risk
    result = _parse("调节亮度")
    assert any("信息不完整" in r for r in result.risk_points)
