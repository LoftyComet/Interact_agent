from __future__ import annotations

import hashlib
from dataclasses import dataclass

from gesture_agent.core.models import SourceChunk


@dataclass(frozen=True)
class EvidenceRecord:
    citation_id: int
    chunk_id: str
    title: str
    citation: str
    layer: str
    score: float
    excerpt: str
    excerpt_sha256: str

    def render(self) -> str:
        return (
            f"[{self.citation_id}] {self.title}\n"
            f"来源：{self.citation}；索引块：{self.chunk_id}；"
            f"原文指纹：{self.excerpt_sha256[:12]}；层级：{self.layer}；匹配分：{self.score}\n"
            f"{self.excerpt}"
        )


@dataclass(frozen=True)
class EvidencePacket:
    """A read-only, fully traceable view over retrieved corpus chunks."""

    records: tuple[EvidenceRecord, ...]

    def render(self) -> str:
        return "\n\n".join(record.render() for record in self.records)


def build_evidence_packet(
    chunks: list[SourceChunk],
    *,
    max_chars: int = 3600,
) -> EvidencePacket:
    records = []
    for index, chunk in enumerate(chunks, start=1):
        excerpt = truncate_verbatim(chunk.text, max_chars)
        if excerpt not in chunk.text:
            raise ValueError(f"Evidence excerpt is not verbatim for chunk {chunk.id}")
        records.append(EvidenceRecord(
            citation_id=index,
            chunk_id=chunk.id,
            title=chunk.title,
            citation=chunk.citation(),
            layer=chunk.layer,
            score=chunk.score,
            excerpt=excerpt,
            excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        ))
    return EvidencePacket(records=tuple(records))


def truncate_verbatim(text: str, limit: int) -> str:
    """Return an unchanged prefix ending at a source sentence boundary."""

    if len(text) <= limit:
        return text
    for separator in ("。", "；", "\n", ".", ";"):
        position = text.rfind(separator, 0, limit)
        if position >= limit // 2:
            return text[: position + 1]
    return text[:limit]
