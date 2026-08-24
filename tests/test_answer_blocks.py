from gesture_agent.verification import is_limitation_only_corpus, parse_answer_blocks


def test_unmarked_answer_is_legacy_corpus_evidence() -> None:
    document = parse_answer_blocks("结论。[1]\n\n## 核心定义\n\n定义。[1]")

    assert document.explicit_markers is False
    assert [block.type for block in document.blocks] == ["corpus_evidence"]
    assert document.blocks[0].citation_ids == (1,)
    assert document.render_markdown().startswith("结论")


def test_marked_answer_separates_evidence_from_reasoning() -> None:
    source = """<!-- ixdl-answer-block:corpus_evidence -->

书中结论。[1]

## 资料依据

书中案例。[2]

<!-- ixdl-answer-block:design_reasoning -->

## 设计建议

可以尝试放大控件。"""
    document = parse_answer_blocks(source)

    assert [block.type for block in document.blocks] == ["corpus_evidence", "design_reasoning"]
    assert document.corpus_markdown.endswith("书中案例。[2]")
    assert document.blocks[0].citation_ids == (1, 2)
    assert document.blocks[1].citation_ids == ()
    assert document.to_list()[1]["title"] == "设计推导（仅供参考）"


def test_replacing_corpus_preserves_design_reasoning() -> None:
    source = """<!-- ixdl-answer-block:corpus_evidence -->
旧的无依据结论。
<!-- ixdl-answer-block:design_reasoning -->
可以尝试另一种布局。"""
    document = parse_answer_blocks(source)

    rendered = document.replace_corpus_markdown("安全结论。[1]").render_markdown()

    assert "旧的无依据结论" not in rendered
    assert "安全结论。[1]" in rendered
    assert "可以尝试另一种布局" in rendered


def test_limitation_only_detection_does_not_hide_mixed_unsupported_claims() -> None:
    assert is_limitation_only_corpus("当前资料不足以支持更具体的结论。")
    assert is_limitation_only_corpus(
        "## 修改建议\n\n当前资料没有直接证据，无法在不进行额外推导的情况下展开这一部分。"
    )
    assert not is_limitation_only_corpus(
        "当前资料不足，但这个方案一定能减少误触。"
    )
