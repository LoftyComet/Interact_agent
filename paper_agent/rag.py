"""论文 RAG 查询管线。

question -> embed(query) -> 向量 top-k -> 拼 prompt -> chat / chat_stream ->
带 [n] 行内引用的回答。不复用词典的意图/输出框架机制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from gesture_agent.providers.siliconflow import SiliconFlowClient
from .corpus import load_corpus
from .index import VectorIndex
from .models import Paper, RetrievedPaper

SYSTEM_PROMPT = (
    "You are an HCI research assistant. Answer the user's question using ONLY the "
    "provided papers (title, authors, year, venue, abstract). Cite sources inline as "
    "[1], [2] matching the numbered list. If the papers do not cover the question, say "
    "so plainly rather than inventing facts. Answer in the same language as the question."
)


class PaperRAG:
    def __init__(self, corpus: list[Paper], index: VectorIndex, client: SiliconFlowClient):
        self.client = client
        self.index = index
        self._by_id: dict[str, Paper] = {p.id: p for p in corpus}

    @classmethod
    def load(cls, data_dir: str | Path, *, timeout: Optional[int] = None) -> "PaperRAG":
        """从语料 + 向量索引目录加载,并构造 SiliconFlow client。"""
        data_dir = Path(data_dir)
        corpus = load_corpus(data_dir / "corpus.jsonl")
        index = VectorIndex.load(data_dir)
        client = SiliconFlowClient.from_env(timeout=timeout)
        return cls(corpus, index, client)

    def retrieve(self, question: str, k: int = 6) -> list[RetrievedPaper]:
        """向量检索,返回 top-k 命中论文。"""
        query_vec = self.client.embed([question])
        if not query_vec:
            return []
        hits = self.index.top_k(np.asarray(query_vec[0], dtype=np.float32), k)
        results: list[RetrievedPaper] = []
        for paper_id, score in hits:
            paper = self._by_id.get(paper_id)
            if paper is not None:
                results.append(RetrievedPaper(paper=paper, score=score))
        return results

    def build_messages(self, question: str, retrieved: list[RetrievedPaper]) -> list[dict]:
        """根据命中论文拼装 chat messages。"""
        lines: list[str] = []
        for idx, item in enumerate(retrieved, start=1):
            p = item.paper
            authors = ", ".join(p.authors[:4]) + (" et al." if len(p.authors) > 4 else "")
            header = f"[{idx}] {p.title} — {authors} ({p.year}, {p.venue}) [{p.citations} citations]"
            abstract = p.abstract or "(no abstract available)"
            lines.append(f"{header}\n{abstract}")
        context = "\n\n".join(lines) if lines else "(no papers retrieved)"
        user = f"Question: {question}\n\nRetrieved papers:\n{context}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def references(retrieved: list[RetrievedPaper]) -> str:
        """把命中论文整理成 References 区块,[n] 映射到引用 + doi/url。"""
        if not retrieved:
            return ""
        lines = ["", "---", "**References**", ""]
        for idx, item in enumerate(retrieved, start=1):
            p = item.paper
            link = p.url or (f"https://doi.org/{p.doi}" if p.doi else "")
            suffix = f" {link}" if link else ""
            lines.append(f"[{idx}] {p.citation()}{suffix}")
        return "\n".join(lines)

    def answer(
        self,
        question: str,
        k: int = 6,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> tuple[str, list[RetrievedPaper]]:
        """非流式:返回(回答 + References, 命中论文列表)。"""
        retrieved = self.retrieve(question, k)
        messages = self.build_messages(question, retrieved)
        body = self.client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return body + "\n" + self.references(retrieved), retrieved

    def answer_stream(
        self,
        question: str,
        k: int = 6,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> tuple[Iterator[str], list[RetrievedPaper]]:
        """流式:返回(delta 迭代器, 命中论文列表)。References 由调用方在结束后追加。"""
        retrieved = self.retrieve(question, k)
        messages = self.build_messages(question, retrieved)
        stream = self.client.chat_stream(messages, temperature=temperature, max_tokens=max_tokens)
        return stream, retrieved
