from __future__ import annotations

import numpy as np

from gesture_agent.indexing.indexes import ChunkVectorIndex, LexicalIndex


def _chunk(chunk_id: str, text: str, digest: str, authority: int = 100) -> dict:
    return {
        "id": chunk_id,
        "document_id": "doc",
        "file_id": "file",
        "title": text[:8],
        "heading_path": [text[:8]],
        "text": text,
        "retrieval_text": text,
        "source_path": "book.docx",
        "source_locators": [{"paragraph": 1}],
        "role": "primary",
        "authority": authority,
        "content_sha256": digest,
        "previous_chunk_id": "",
        "next_chunk_id": "",
    }


def test_lexical_index_supports_chinese_bigrams_and_term_boost(tmp_path) -> None:
    chunks = [
        _chunk("single", "单击是一种离散交互机制", "a"),
        _chunk("drag", "拖拽用于连续位置控制", "b"),
    ]
    index = LexicalIndex.build(tmp_path / "knowledge.sqlite", chunks, {"single": ["单击"]})

    results = index.search("什么是单击", limit=2)

    assert results[0]["id"] == "single"
    assert results[0]["terms"] == ["单击"]
    assert index.get("drag")["text"] == "拖拽用于连续位置控制"


class _FakeEmbedder:
    embedding_model = "fake-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


def test_vector_index_reuses_unchanged_content_hashes(tmp_path) -> None:
    first_embedder = _FakeEmbedder()
    chunks = [_chunk("a", "单击", "same"), _chunk("b", "拖拽", "old")]
    first = ChunkVectorIndex.build(chunks, first_embedder)
    first.save(tmp_path)

    second_embedder = _FakeEmbedder()
    changed = [_chunk("a-new", "单击", "same"), _chunk("b-new", "长按", "new")]
    second = ChunkVectorIndex.build(changed, second_embedder, existing=ChunkVectorIndex.load(tmp_path))

    assert second_embedder.calls == [["长按"]]
    assert second.rows[0]["chunk_id"] == "a-new"
    assert np.allclose(second.vectors[0], first.vectors[0])
    assert second.top_k(second.vectors[0], 1)[0][0] == "a-new"
