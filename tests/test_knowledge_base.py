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


def test_term_inventory_config_can_load_aliases(tmp_path) -> None:
    config = tmp_path / "term_inventory.json"
    config.write_text(
        """
{
  "mode": "merge",
  "aliases": {
    "点按": "单击"
  }
}
""".strip(),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load("data", term_inventory_path=config)

    assert kb.term_inventory.aliases["点按"] == "单击"
    assert "单击" in kb.find_terms("点按是什么？")


def test_alias_mapping_normalizes_user_query() -> None:
    kb = KnowledgeBase.load("data")

    assert "单击" in kb.find_terms("点一下是什么？")
    assert "长按" in kb.find_terms("按住有什么风险？")
    assert "点一下" not in kb.find_terms("点一下是什么？")
    assert kb.normalize_query("点一下和按住有什么区别？") == "单击和长按有什么区别?"


def test_query_rewrite_expands_structured_terms() -> None:
    kb = KnowledgeBase.load("data")

    rewritten = kb.rewrite_query("点一下是什么？")

    assert "单击" in rewritten
    assert "基础交互机制" in rewritten


def test_structured_knowledge_can_be_loaded_from_json(tmp_path) -> None:
    structured = tmp_path / "structured_knowledge.json"
    structured.write_text(
        """
{
  "items": [
    {
      "id": "custom:air-tap",
      "term": "空中点按",
      "term_type": "基础交互机制",
      "layer": "interaction_mechanism",
      "definition": "在空中完成一次短促点按。",
      "aliases": ["air tap"],
      "properties": ["位置属性"],
      "mechanisms": ["单击"],
      "control_forms": ["手势"],
      "response_logic": "系统在识别短促点按后触发一次确认。",
      "related_terms": ["单击"]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    kb = KnowledgeBase.load("data", structured_knowledge_path=structured)

    assert any(item.term == "空中点按" for item in kb.structured_items)
    assert "air tap" in kb.term_inventory.aliases
    assert kb.term_inventory.aliases["air tap"] == "空中点按"
    assert "空中点按" in kb.find_terms("air tap 怎么设计？")


def test_hybrid_search_uses_alias_and_structured_rerank() -> None:
    kb = KnowledgeBase.load("data")

    results = kb.search("点一下是什么？", top_k=3)

    assert results
    assert "单击" in results[0].title


def test_export_structured_knowledge(tmp_path) -> None:
    kb = KnowledgeBase.load("data")
    output = tmp_path / "structured.json"

    kb.export_structured_knowledge(output)

    text = output.read_text(encoding="utf-8")
    assert '"items"' in text
    assert "单击" in text
