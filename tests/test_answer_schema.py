from __future__ import annotations

from gesture_agent.verification import parse_answer_markdown, render_answer_markdown


FRAME = ["核心定义", "基础属性"]


def test_parse_and_render_in_frame_order() -> None:
    source = "直接结论。[1]\n\n## 基础属性\n\n属性内容。\n\n## 核心定义（概览）\n\n定义内容。"
    result = parse_answer_markdown(source, FRAME)

    assert result.valid
    assert result.document.direct_answer == "直接结论。[1]"
    assert result.document.citation_ids == (1,)
    assert render_answer_markdown(result.document, FRAME) == (
        "直接结论。[1]\n\n"
        "## 核心定义\n\n定义内容。\n\n"
        "## 基础属性\n\n属性内容。"
    )


def test_reports_missing_direct_answer_duplicate_and_empty_section() -> None:
    source = "## 核心定义\n内容\n## 核心定义\n重复\n## 基础属性\n"
    result = parse_answer_markdown(source, FRAME)
    codes = {violation.code for violation in result.violations}

    assert "missing_direct_answer" in codes
    assert "duplicate_section" in codes
    assert "empty_section" in codes


def test_rejects_whole_code_fence_and_json() -> None:
    fenced = parse_answer_markdown("```markdown\n结论\n## 核心定义\n内容\n```", ["核心定义"])
    json_result = parse_answer_markdown('{"answer": "内容"}', FRAME)

    assert any(v.code == "format_error" for v in fenced.violations)
    assert any(v.code == "format_error" for v in json_result.violations)


def test_block_marker_is_not_a_direct_answer() -> None:
    result = parse_answer_markdown(
        "<!-- ixdl-answer-block:corpus_evidence -->\n\n## 核心定义\n\n内容足够长。",
        ["核心定义"],
    )

    assert any(v.code == "missing_direct_answer" for v in result.violations)


def test_lead_in_without_content_is_an_empty_section() -> None:
    result = parse_answer_markdown(
        "直接结论。\n\n## 核心定义\n\n以下内容需要确认：",
        ["核心定义"],
    )

    assert any(v.code == "empty_section" for v in result.violations)
