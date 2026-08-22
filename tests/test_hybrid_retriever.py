from __future__ import annotations

import numpy as np

from gesture_agent.indexing.indexes import ChunkVectorIndex, LexicalIndex
from gesture_agent.retrieval import HybridRetriever


def _chunk(chunk_id: str, title: str, text: str, digest: str) -> dict:
    return {
        "id": chunk_id,
        "document_id": "doc",
        "file_id": "file",
        "title": title,
        "heading_path": [title],
        "text": text,
        "retrieval_text": f"{title}\n{text}",
        "source_path": "book.docx",
        "source_locators": [{"paragraph": 1}],
        "role": "primary",
        "authority": 100,
        "content_sha256": digest,
        "previous_chunk_id": "",
        "next_chunk_id": "",
    }


class _QueryEmbedder:
    embedding_model = "fake-v1"

    def embed(self, texts, *, batch_size=32):
        return [[1.0, 0.0] for _ in texts]


def test_retriever_normalizes_alias_and_returns_source_context(tmp_path) -> None:
    chunks = [
        _chunk("click", "单击", "单击是一次短促的离散触发。", "a"),
        _chunk("drag", "拖拽", "拖拽用于连续位置控制。", "b"),
    ]
    lexical = LexicalIndex.build(tmp_path / "knowledge.sqlite", chunks, {"click": ["单击"]})
    retriever = HybridRetriever(lexical, aliases={"点一下": "单击"}, canonical_terms=["单击", "拖拽"])

    results = retriever.retrieve("点一下是什么", top_k=1, expand_neighbors=0)

    assert retriever.mode == "lexical"
    assert results[0].chunk_id == "click"
    assert results[0].channels == ["lexical"]
    assert results[0].source_locators == [{"paragraph": 1}]


def test_retriever_combines_vector_and_lexical_ranks(tmp_path) -> None:
    chunks = [
        _chunk("click", "单击", "单击是离散触发。", "a"),
        _chunk("drag", "拖拽", "拖拽是连续控制。", "b"),
    ]
    lexical = LexicalIndex.build(tmp_path / "knowledge.sqlite", chunks, {})
    vectors = ChunkVectorIndex(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [{"chunk_id": "click", "content_sha256": "a"}, {"chunk_id": "drag", "content_sha256": "b"}],
        "fake-v1",
    )
    retriever = HybridRetriever(lexical, vectors=vectors, embedder=_QueryEmbedder())

    results = retriever.retrieve("单击", top_k=1, expand_neighbors=0)

    assert retriever.mode == "hybrid"
    assert results[0].chunk_id == "click"
    assert results[0].channels == ["lexical", "vector"]
