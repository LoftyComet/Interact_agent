import json

import pytest

from gesture_agent.cli import main
from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser
from gesture_agent.learning.output_frames import load_output_frames


def test_output_frames_config_can_merge_custom_frame(tmp_path) -> None:
    config = tmp_path / "output_frames.json"
    config.write_text(
        json.dumps(
            {
                "mode": "merge",
                "frames": {
                    "design_evaluation": ["专家复述", "专家诊断", "专家建议"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load("data")
    output_frames = load_output_frames("data", output_frames_path=config)
    parser = QuestionParser(kb, output_frames=output_frames)
    structure = parser.parse("请评估这个设计方案：用户长按音量旋钮后拖动来调节音量。")

    assert structure.output_frame == ["专家复述", "专家诊断", "专家建议"]
    assert parser.output_frames.frames["interaction_compare"] == ["对比对象", "共同基础", "核心差异", "适用边界", "选择建议"]


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


def test_cli_can_load_custom_output_frames(tmp_path, capsys) -> None:
    config = tmp_path / "output_frames.json"
    config.write_text(
        json.dumps(
            {
                "mode": "merge",
                "frames": {
                    "interaction_compare": ["专家对比对象", "专家核心差异"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["单击和长按有什么区别？", "--dry-run", "--output-frames", str(config)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "专家对比对象" in captured.out
    assert "专家核心差异" in captured.out
