from gesture_agent.knowledge import KnowledgeBase


def test_kb_loads_core_chunks() -> None:
    kb = KnowledgeBase.load("data")

    assert len(kb.chunks) > 30
    assert "单击" in kb.terms
    assert "长按" in kb.terms


def test_search_returns_relevant_chunk() -> None:
    kb = KnowledgeBase.load("data")

    results = kb.search("长按是什么？", top_k=3)

    assert results
    assert any("长按" in chunk.title for chunk in results)


def test_comparison_document_is_chunked_and_searchable() -> None:
    kb = KnowledgeBase.load("data")

    comparison_chunks = [chunk for chunk in kb.chunks if chunk.source.endswith("交互机制对比.md")]
    results = kb.search("快击和点击缓冲有什么区别？", top_k=5)

    assert len(comparison_chunks) >= 6
    assert any("快击" in chunk.terms and "点击缓冲" in chunk.terms for chunk in comparison_chunks)
    assert any(chunk.source.endswith("交互机制对比.md") and "快击" in chunk.title for chunk in results)
    assert all(chunk.layer == "interaction_mechanism" for chunk in comparison_chunks)


def test_term_inventory_groups_canonical_terms() -> None:
    kb = KnowledgeBase.load("data")
    inventory = kb.term_inventory

    assert "单击" in inventory.by_layer["interaction_mechanism"]
    assert "长按拖拽" in inventory.by_layer["interaction_mechanism"]
    assert "旋钮" in inventory.by_layer["control_form"]
    assert "二元属性" in inventory.by_layer["basic_property"]
    assert "交互机制" in inventory.structural_terms
    assert "1-a" not in inventory.by_layer["interaction_mechanism"]
    assert "a 开关" not in inventory.by_layer["interaction_mechanism"]


def test_term_inventory_config_can_merge_custom_terms(tmp_path) -> None:
    config = tmp_path / "term_inventory.json"
    config.write_text(
        """
{
  "mode": "merge",
  "by_layer": {
    "interaction_mechanism": ["自定义机制"],
    "control_form": ["自定义控件"]
  }
}
""".strip(),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load("data", term_inventory_path=config)

    assert kb.term_inventory.source.endswith("term_inventory.json")
    assert kb.term_inventory.by_layer["interaction_mechanism"][0] == "自定义机制"
    assert "单击" in kb.term_inventory.by_layer["interaction_mechanism"]
    assert "自定义控件" in kb.term_inventory.by_layer["control_form"]


def test_term_inventory_config_can_replace_generated_terms(tmp_path) -> None:
    config = tmp_path / "term_inventory.json"
    config.write_text(
        """
{
  "mode": "replace",
  "structural_terms": ["结构词"],
  "by_layer": {
    "interaction_mechanism": ["自定义机制"]
  }
}
""".strip(),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load("data", term_inventory_path=config)

    assert kb.term_inventory.structural_terms == ["结构词"]
    assert kb.term_inventory.by_layer["interaction_mechanism"] == ["自定义机制"]
    assert "单击" not in kb.term_inventory.by_layer["interaction_mechanism"]


def test_term_inventory_config_can_load_expert_term_types(tmp_path) -> None:
    config = tmp_path / "term_inventory.json"
    config.write_text(
        """
{
  "mode": "merge",
  "by_type": {
    "基础交互机制": ["专家机制"],
    "响应类型": ["微变"]
  }
}
""".strip(),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load("data", term_inventory_path=config)

    assert kb.term_inventory.by_type["基础交互机制"] == ["专家机制"]
    assert kb.term_inventory.by_type["响应类型"] == ["微变"]
    assert "专家机制" in kb.terms
