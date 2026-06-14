"""论文 RAG 子系统的核心数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    """一篇论文的元数据 + 摘要。

    id 使用 OpenAlex work id 作为稳定主键;全文不抓取,仅保留摘要。
    """

    id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    abstract: str
    doi: str = ""
    url: str = ""
    citations: int = 0
    concepts: list[str] = field(default_factory=list)

    def citation(self) -> str:
        """生成简短的引用字符串:'Author et al. (Year). Title. Venue.'"""
        if not self.authors:
            author = ""
        elif len(self.authors) > 1:
            author = f"{self.authors[0]} et al."
        else:
            author = self.authors[0]
        year = self.year if self.year is not None else "n.d."
        parts = [p for p in [author, f"({year})."] if p]
        head = " ".join(parts).strip()
        tail = f"{self.title}. {self.venue}.".strip()
        return f"{head} {tail}".strip()

    def embed_text(self) -> str:
        """用于嵌入的文本:标题 + 摘要一起编码。"""
        if self.abstract:
            return f"{self.title}\n\n{self.abstract}"
        return self.title


@dataclass
class RetrievedPaper:
    """检索命中的论文 + 余弦相似度得分。"""

    paper: Paper
    score: float
