from __future__ import annotations

import numpy as np

from gesture_agent.indexing.indexes import ChunkVectorIndex, LexicalIndex
from gesture_agent.retrieval import HybridRetriever


def _chunk(
    chunk_id: str,
    title: str,
    text: str,
    digest: str,
    *,
    role: str = "primary",
    authority: int = 100,
) -> dict:
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
        "role": role,
        "authority": authority,
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


def test_retriever_reserves_primary_candidates_for_recognized_terms(tmp_path) -> None:
    chunks = [
        _chunk(
            f"secondary-{index}",
            "旋钮操作",
            f"背景资料中的旋钮讨论 {index}",
            f"secondary-{index}",
            role="secondary",
            authority=60,
        )
        for index in range(35)
    ]
    primary = _chunk("primary-knob", "控件定义", "旋钮是一种控件形态。", "primary")
    primary["heading_path"] = ["旋钮", "控件定义"]
    chunks.append(primary)
    lexical = LexicalIndex.build(
        tmp_path / "knowledge.sqlite",
        chunks,
        {chunk["id"]: ["旋钮"] for chunk in chunks},
    )
    retriever = HybridRetriever(lexical, canonical_terms=["旋钮"])

    results = retriever.retrieve("旋钮为什么适合小屏幕", top_k=6, recall_k=30, expand_neighbors=0)

    assert any(result.chunk_id == "primary-knob" for result in results)


def test_retriever_does_not_reserve_primary_candidates_without_domain_terms(tmp_path) -> None:
    chunks = [
        _chunk("secondary", "设计反馈", "请减少回答文字。", "secondary", role="secondary", authority=60),
        _chunk("primary", "信息组织", "正文内容。", "primary"),
    ]
    lexical = LexicalIndex.build(tmp_path / "knowledge.sqlite", chunks, {})
    retriever = HybridRetriever(lexical, canonical_terms=["旋钮"])

    results = retriever.retrieve("请减少回答文字", top_k=1, expand_neighbors=0)

    assert results[0].chunk_id == "secondary"


def test_retriever_uses_authoritative_source_for_duplicate_content(tmp_path) -> None:
    chunks = [
        _chunk(
            "secondary",
            "旋钮操作",
            "旋钮适合连续调节。",
            "same-content",
            role="secondary",
            authority=60,
        ),
        _chunk("primary", "控件定义", "旋钮适合连续调节。", "same-content"),
    ]
    chunks[1]["heading_path"] = ["旋钮", "控件定义"]
    lexical = LexicalIndex.build(
        tmp_path / "knowledge.sqlite",
        chunks,
        {"secondary": ["旋钮"], "primary": ["旋钮"]},
    )
    retriever = HybridRetriever(lexical, canonical_terms=["旋钮"])

    results = retriever.retrieve("旋钮操作", top_k=2, expand_neighbors=0)

    assert [result.chunk_id for result in results] == ["primary"]


def test_closest_heading_match_beats_shared_distant_ancestor(tmp_path) -> None:
    direct = _chunk("direct", "语音交互", "语音定义。", "direct")
    distant = _chunk("distant", "无关案例", "语音案例背景。", "distant")
    distant["heading_path"] = ["语音交互", "创新案例", "无关案例"]
    lexical = LexicalIndex.build(
        tmp_path / "knowledge.sqlite",
        [distant, direct],
        {"direct": ["语音"], "distant": ["语音"]},
    )
    retriever = HybridRetriever(lexical, canonical_terms=["语音"])

    results = retriever.retrieve("语音", top_k=2, expand_neighbors=0)

    assert results[0].chunk_id == "direct"


def test_terminology_channel_recovers_exact_multi_term_comparison_from_noisy_query(tmp_path) -> None:
    target = _chunk(
        "comparison",
        "单击 VS 按下",
        "单击在抬起后触发，按下在状态变化时触发。",
        "comparison",
    )
    noise = _chunk(
        "noise",
        "常见问题",
        "书里讲了很多内容，什么时候使用还是容易糊涂。",
        "noise",
    )
    lexical = LexicalIndex.build(
        tmp_path / "knowledge.sqlite",
        [noise, target],
        {"noise": [], "comparison": ["单击", "按下"]},
    )
    retriever = HybridRetriever(lexical, canonical_terms=["单击", "按下"])

    results = retriever.retrieve(
        "书里讲得很糊涂，到底什么时候应该使用它们",
        preferred_terms=["单击", "按下"],
        top_k=1,
        recall_k=1,
        expand_neighbors=0,
    )

    assert results[0].chunk_id == "comparison"
    assert "terminology" in results[0].channels


def test_retrieval_exposes_only_inventory_backed_taxonomy_terms(tmp_path) -> None:
    target = _chunk("click", "单击", "单击是离散触发。", "click")
    lexical = LexicalIndex.build(
        tmp_path / "knowledge.sqlite",
        [target],
        {"click": ["单击"]},
    )
    retriever = HybridRetriever(
        lexical,
        canonical_terms=["单击", "点击类"],
        term_groups={"单击": ["点击类"]},
    )

    result = retriever.retrieve("单击", top_k=1, expand_neighbors=0)[0]

    assert result.terms == ["单击", "点击类"]
