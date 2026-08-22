from __future__ import annotations

from gesture_agent.indexing.chunking import chunk_documents
from gesture_agent.indexing.models import DocumentBlock, NormalizedDocument


def _document(blocks: list[DocumentBlock]) -> NormalizedDocument:
    return NormalizedDocument(
        schema_version="1.0",
        id="doc_test",
        file_id="file_test",
        title="交互机制",
        source_path="book.docx",
        source_sha256="abc",
        source_format=".docx",
        role="primary",
        authority=100,
        blocks=blocks,
    )


def test_chunker_preserves_heading_path_and_source_locator() -> None:
    blocks = [
        DocumentBlock("b1", 0, "heading", "第二篇", 1, {"paragraph": 0}),
        DocumentBlock("b2", 1, "heading", "单击", 2, {"paragraph": 1}),
        DocumentBlock("b3", 2, "paragraph", "单击是一种离散交互机制。", None, {"paragraph": 2}),
    ]

    chunks, report = chunk_documents([_document(blocks)], max_chars=300, overlap_chars=30)

    assert chunks[0].heading_path == ["第二篇", "单击"]
    assert chunks[0].title == "单击"
    assert chunks[0].source_locators == [{"paragraph": 2}]
    assert "交互机制 > 第二篇 > 单击" in chunks[0].retrieval_text
    assert report["chunk_count"] == 1


def test_chunk_ids_are_stable_and_neighbors_are_linked() -> None:
    long_text = "。".join([f"句子{i}" * 12 for i in range(20)]) + "。"
    blocks = [
        DocumentBlock("b1", 0, "heading", "长按", 1, {}),
        DocumentBlock("b2", 1, "paragraph", long_text, None, {"paragraph": 1}),
    ]

    first, _ = chunk_documents([_document(blocks)], max_chars=220, overlap_chars=20)
    second, _ = chunk_documents([_document(blocks)], max_chars=220, overlap_chars=20)

    assert len(first) > 1
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].next_chunk_id == first[1].id
    assert first[1].previous_chunk_id == first[0].id


def test_chunker_rejects_invalid_limits() -> None:
    try:
        chunk_documents([], max_chars=100)
    except ValueError as exc:
        assert "max_chars" in str(exc)
    else:
        raise AssertionError("Expected invalid max_chars to fail")
