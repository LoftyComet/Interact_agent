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
