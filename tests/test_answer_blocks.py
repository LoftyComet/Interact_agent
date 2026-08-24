from gesture_agent.verification import parse_answer_blocks


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
