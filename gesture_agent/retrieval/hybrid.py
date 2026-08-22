"""Hybrid BM25/vector retrieval behind a small runtime interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from gesture_agent.indexing import ChunkVectorIndex, LexicalIndex, verify_manifest
from gesture_agent.indexing.indexes import Embedder


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    title: str
    text: str
    context_text: str
    context_chunk_ids: list[str]
    source_path: str
    source_locators: list[dict]
    heading_path: list[str]
    terms: list[str]
    role: str
    authority: int
    score: float
    channels: list[str]


class HybridRetriever:
    """Retrieve knowledge with lexical-only or lexical+vector adapters.

    Callers use the same interface regardless of which index adapters are
    present. Vector retrieval activates only when both a built vector index and
    a compatible embedder are available.
    """

    def __init__(
        self,
        lexical: LexicalIndex,
        *,
        aliases: Optional[dict[str, str]] = None,
        canonical_terms: Optional[list[str]] = None,
        vectors: Optional[ChunkVectorIndex] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        if vectors is not None and embedder is None:
            raise ValueError("A vector index requires an embedder for query vectors")
        if vectors is not None and embedder is not None and vectors.model != embedder.embedding_model:
            raise ValueError(f"Embedding model mismatch: index={vectors.model}, query={embedder.embedding_model}")
        self.lexical = lexical
        self.aliases = aliases or {}
        self.canonical_terms = canonical_terms or []
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker

    @classmethod
    def from_directory(
        cls,
        index_dir: str | Path,
        *,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        verify: bool = True,
    ) -> "HybridRetriever":
        root = Path(index_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if verify:
            errors = verify_manifest(root, manifest)
            if errors:
                raise ValueError("Invalid knowledge index: " + ", ".join(errors))
        terminology = json.loads((root / "terminology.json").read_text(encoding="utf-8"))
        vectors = None
        if manifest.get("embedding", {}).get("status") == "ready" and embedder is not None:
            vectors = ChunkVectorIndex.load(root / "vectors")
        return cls(
            LexicalIndex(root / "knowledge.sqlite"),
            aliases=dict(terminology.get("aliases", {})),
            canonical_terms=[str(item["label"]) for item in terminology.get("terms", [])],
            vectors=vectors,
            embedder=embedder if vectors is not None else None,
            reranker=reranker,
        )

    @property
    def mode(self) -> str:
        return "hybrid" if self.vectors is not None else "lexical"

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        recall_k: int = 30,
        preferred_terms: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
        expand_neighbors: int = 1,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        normalized = self._normalize_query(query, preferred_terms or [])
        query_terms = self._find_query_terms(normalized, preferred_terms or [])
        lexical_hits = self.lexical.search(normalized, limit=max(recall_k, top_k), roles=roles)
        candidates: dict[str, dict] = {hit["id"]: hit for hit in lexical_hits}
        ranks: dict[str, dict[str, int]] = {}
        for rank, hit in enumerate(lexical_hits, start=1):
            ranks.setdefault(hit["id"], {})["lexical"] = rank

        if self.vectors is not None and self.embedder is not None:
            query_vectors = self.embedder.embed([normalized], batch_size=1)
            if query_vectors:
                vector_hits = self.vectors.top_k(query_vectors[0], limit=max(recall_k, top_k))
                for rank, (chunk_id, _score) in enumerate(vector_hits, start=1):
                    row = candidates.get(chunk_id) or self.lexical.get(chunk_id)
                    if row is not None and (not roles or row["role"] in roles):
                        candidates[chunk_id] = row
                        ranks.setdefault(chunk_id, {})["vector"] = rank

        scored: list[tuple[dict, float, list[str]]] = []
        for chunk_id, row in candidates.items():
            channel_ranks = ranks.get(chunk_id, {})
            score = sum(1.0 / (60 + rank) for rank in channel_ranks.values())
            heading_text = " ".join([row.get("title", ""), *row.get("heading_path", [])])
            score += sum(0.03 for term in query_terms if term in heading_text)
            score += sum(0.008 for term in query_terms if term in row.get("terms", []))
            score *= 1.0 + max(int(row.get("authority", 0)), 0) / 2000.0
            scored.append((row, score, sorted(channel_ranks)))

        if self.reranker and scored:
            rerank_scores = self.reranker.score(normalized, [row["retrieval_text"] for row, _, _ in scored])
            if len(rerank_scores) != len(scored):
                raise ValueError("Reranker returned a different number of scores than candidates")
            scored = [
                (row, score + float(rerank_score), channels + ["reranker"])
                for (row, score, channels), rerank_score in zip(scored, rerank_scores)
            ]

        scored.sort(key=lambda item: (item[1], int(item[0]["authority"])), reverse=True)
        selected = self._dedupe_by_content(scored)[:top_k]
        return [self._to_result(row, score, channels, expand_neighbors) for row, score, channels in selected]

    def _normalize_query(self, query: str, preferred_terms: list[str]) -> str:
        rewritten = query.strip()
        for alias, canonical in sorted(self.aliases.items(), key=lambda item: (-len(item[0]), item[0])):
            if alias and alias in rewritten:
                rewritten = rewritten.replace(alias, canonical)
        additions = [term for term in preferred_terms if term and term not in rewritten]
        return " ".join([rewritten, *additions]).strip()

    def _find_query_terms(self, query: str, preferred_terms: list[str]) -> list[str]:
        found = [term for term in preferred_terms if term]
        for term in sorted(self.canonical_terms, key=lambda value: (-len(value), value)):
            if term and term in query and term not in found:
                found.append(term)
        return found

    @staticmethod
    def _dedupe_by_content(scored: list[tuple[dict, float, list[str]]]) -> list[tuple[dict, float, list[str]]]:
        result: list[tuple[dict, float, list[str]]] = []
        seen: set[str] = set()
        for item in scored:
            digest = str(item[0].get("content_sha256", "")) or item[0]["id"]
            if digest not in seen:
                seen.add(digest)
                result.append(item)
        return result

    def _to_result(self, row: dict, score: float, channels: list[str], expand_neighbors: int) -> RetrievalResult:
        context_rows: list[dict] = []
        if expand_neighbors > 0 and len(row["text"]) < 350:
            previous_id = row.get("previous_chunk_id", "")
            next_id = row.get("next_chunk_id", "")
            if previous_id:
                previous = self.lexical.get(previous_id)
                if previous and previous["document_id"] == row["document_id"]:
                    context_rows.append(previous)
            context_rows.append(row)
            if next_id:
                following = self.lexical.get(next_id)
                if following and following["document_id"] == row["document_id"]:
                    context_rows.append(following)
        else:
            context_rows = [row]
        return RetrievalResult(
            chunk_id=row["id"],
            title=row["title"],
            text=row["text"],
            context_text="\n\n".join(item["text"] for item in context_rows),
            context_chunk_ids=[item["id"] for item in context_rows],
            source_path=row["source_path"],
            source_locators=row["source_locators"],
            heading_path=row["heading_path"],
            terms=row["terms"],
            role=row["role"],
            authority=int(row["authority"]),
            score=round(score, 8),
            channels=channels,
        )
