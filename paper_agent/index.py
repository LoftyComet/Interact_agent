"""向量索引:嵌入向量的磁盘存取、增量构建与余弦 top-k 检索。

磁盘格式(均位于同一目录):
- embeddings.npy   float32 [N, D],构建时 L2 归一化(余弦 = 点积)
- index.jsonl      N 行,行对齐:{"row": i, "id": paper_id}
- index.meta.json  {"model": ..., "dim": D, "count": N}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from .models import Paper

if TYPE_CHECKING:
    from gesture_agent.providers.siliconflow import SiliconFlowClient

EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "index.jsonl"
META_FILE = "index.meta.json"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """对每行做 L2 归一化,零向量保持为零。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class VectorIndex:
    """论文向量索引。vectors 已 L2 归一化,与 ids 按行对齐。"""

    def __init__(self, vectors: np.ndarray, ids: list[str], model: str):
        if vectors.shape[0] != len(ids):
            raise ValueError(f"向量行数 {vectors.shape[0]} 与 id 数 {len(ids)} 不一致")
        self.vectors = vectors.astype(np.float32)
        self.ids = ids
        self.model = model

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, dir_path: str | Path) -> "VectorIndex":
        """从目录加载索引。缺文件时抛 FileNotFoundError。"""
        dir_path = Path(dir_path)
        emb_path = dir_path / EMBEDDINGS_FILE
        idx_path = dir_path / INDEX_FILE
        meta_path = dir_path / META_FILE
        if not emb_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"向量索引不存在:{emb_path} 或 {idx_path}。请先运行 `embed` 子命令。"
            )
        vectors = np.load(emb_path)
        ids: list[str] = []
        for raw_line in idx_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["id"])
        model = ""
        if meta_path.exists():
            model = json.loads(meta_path.read_text(encoding="utf-8")).get("model", "")
        return cls(vectors, ids, model)

    def save(self, dir_path: str | Path) -> None:
        """把向量、对齐清单、元数据写入目录。"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / EMBEDDINGS_FILE, self.vectors)
        with (dir_path / INDEX_FILE).open("w", encoding="utf-8") as fh:
            for row, paper_id in enumerate(self.ids):
                fh.write(json.dumps({"row": row, "id": paper_id}, ensure_ascii=False) + "\n")
        meta = {"model": self.model, "dim": self.dim, "count": len(self.ids)}
        (dir_path / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def build(
        papers: list[Paper],
        client: "SiliconFlowClient",
        existing: Optional["VectorIndex"] = None,
        *,
        batch_size: int = 32,
    ) -> "VectorIndex":
        """构建索引。existing 提供时做增量:只嵌入新增 id。

        若 existing 的模型与 client.embedding_model 不一致,则全量重建,
        避免不同模型的向量混在同一空间。
        """
        model = client.embedding_model
        reuse = existing is not None and existing.model == model
        cached: dict[str, np.ndarray] = {}
        if reuse and existing is not None:
            cached = {pid: existing.vectors[row] for row, pid in enumerate(existing.ids)}

        todo = [p for p in papers if p.id not in cached]
        if todo:
            new_vectors = client.embed([p.embed_text() for p in todo], batch_size=batch_size)
            new_matrix = _normalize(np.asarray(new_vectors, dtype=np.float32))
            for paper, vec in zip(todo, new_matrix):
                cached[paper.id] = vec

        ordered_ids = [p.id for p in papers]
        if not ordered_ids:
            return VectorIndex(np.zeros((0, 0), dtype=np.float32), [], model)
        matrix = np.vstack([cached[pid] for pid in ordered_ids]).astype(np.float32)
        return VectorIndex(matrix, ordered_ids, model)

    def top_k(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """返回与 query 余弦相似度最高的 k 个 (id, score),降序。"""
        if len(self.ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        sims = self.vectors @ q
        k = min(k, sims.shape[0])
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self.ids[i], float(sims[i])) for i in top_idx]
