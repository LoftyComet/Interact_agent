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
