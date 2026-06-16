"""向量索引:top-k、save/load、增量构建测试。"""

import numpy as np

from paper_agent.index import VectorIndex
from paper_agent.models import Paper


def _paper(pid: str) -> Paper:
    return Paper(id=pid, title=f"title {pid}", authors=[], year=2023, venue="UIST", abstract=f"abstract {pid}")


def test_top_k_orders_by_cosine() -> None:
    # 三个正交方向 + 一个偏向 x 的向量
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    index = VectorIndex(vectors, ["x", "y", "z"], "m")
    hits = index.top_k(np.array([0.9, 0.1, 0.0], dtype=np.float32), k=2)
    assert [pid for pid, _ in hits] == ["x", "y"]
    assert hits[0][1] > hits[1][1]


def test_top_k_empty_index() -> None:
    index = VectorIndex(np.zeros((0, 0), dtype=np.float32), [], "m")
    assert index.top_k(np.array([1.0]), k=3) == []


def test_save_load_roundtrip(tmp_path) -> None:
    vectors = _normalized(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    index = VectorIndex(vectors, ["a", "b"], "Pro/BAAI/bge-m3")
    index.save(tmp_path)

    loaded = VectorIndex.load(tmp_path)
    assert loaded.ids == ["a", "b"]
    assert loaded.model == "Pro/BAAI/bge-m3"
    assert loaded.dim == 3
    np.testing.assert_allclose(loaded.vectors, vectors, rtol=1e-5)


class _FakeClient:
    """模拟 SiliconFlowClient.embed:按文本长度生成确定向量。"""

    embedding_model = "Pro/BAAI/bge-m3"

    def __init__(self):
        self.calls = []

    def embed(self, texts, *, batch_size=32):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_build_then_incremental_only_embeds_new() -> None:
    client = _FakeClient()
    papers = [_paper("a"), _paper("b")]
    index = VectorIndex.build(papers, client)
    assert len(index) == 2
    assert len(client.calls[0]) == 2  # 首次嵌入 2 篇

    # 增量:新增 c,a/b 复用
    client.calls.clear()
    papers2 = papers + [_paper("c")]
    index2 = VectorIndex.build(papers2, client, existing=index)
    assert len(index2) == 3
    assert client.calls == [[_paper("c").embed_text()]]  # 只嵌入 c


def test_build_rebuilds_on_model_mismatch() -> None:
    old = VectorIndex(_normalized(np.array([[1, 0, 0]], dtype=np.float32)), ["a"], "old-model")
    client = _FakeClient()  # 模型为 Pro/BAAI/bge-m3,与 old 不符
    index = VectorIndex.build([_paper("a")], client, existing=old)
    assert index.model == "Pro/BAAI/bge-m3"
    assert len(client.calls[0]) == 1  # 全量重嵌


def _normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / norms).astype(np.float32)
