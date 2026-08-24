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


PRIMARY_ROLE = "primary"
PRIMARY_RECALL_LIMIT = 30
PRIMARY_RESULT_SCORE_RATIO = 0.65


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
        term_groups: Optional[dict[str, list[str]]] = None,
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
        self.term_groups = term_groups or {}
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
        canonical_terms = [str(item["label"]) for item in terminology.get("terms", [])]
        canonical_term_set = set(canonical_terms)
        return cls(
            LexicalIndex(root / "knowledge.sqlite"),
            aliases=dict(terminology.get("aliases", {})),
            canonical_terms=canonical_terms,
            term_groups={
                str(item["label"]): [
                    str(group)
                    for group in item.get("group_path", [])
                    if str(group) in canonical_term_set
                ]
                for item in terminology.get("terms", [])
            },
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
        candidate_channels: dict[str, set[str]] = {}
        for rank, hit in enumerate(lexical_hits, start=1):
            ranks.setdefault(hit["id"], {})["lexical"] = rank
            candidate_channels.setdefault(hit["id"], set()).add("lexical")

        # Natural-language questions can bury short canonical terms under BM25
        # query noise. Add a terminology-only recall channel without changing
        # or enriching any corpus text.
        if len(query_terms) >= 2:
            terminology_hits = self.lexical.search(
                " ".join(query_terms),
                limit=max(top_k, min(recall_k, PRIMARY_RECALL_LIMIT)),
                roles=roles,
            )
            for hit in terminology_hits:
                candidates[hit["id"]] = hit
                # This channel exists to expand the candidate set. It carries no
                # reciprocal-rank weight: exact headings and indexed terminology
                # decide whether the recovered row is strong enough to surface.
                candidate_channels.setdefault(hit["id"], set()).add("terminology")

        # A large secondary corpus can otherwise occupy the entire global BM25
        # recall window.  When the parser has recognized domain terminology,
        # reserve a small candidate pool from the canonical handbook.  These are
        # still scored normally, so an unrelated primary chunk cannot displace a
        # substantially more relevant secondary result merely because of role.
        if query_terms and (roles is None or PRIMARY_ROLE in roles):
            primary_hits = self.lexical.search(
                normalized,
                limit=max(top_k, min(recall_k, PRIMARY_RECALL_LIMIT)),
                roles=[PRIMARY_ROLE],
            )
            for rank, hit in enumerate(primary_hits, start=1):
                candidates[hit["id"]] = hit
                channel_ranks = ranks.setdefault(hit["id"], {})
                channel_ranks["lexical"] = min(channel_ranks.get("lexical", rank), rank)
                candidate_channels.setdefault(hit["id"], set()).add("lexical")

        if self.vectors is not None and self.embedder is not None:
            query_vectors = self.embedder.embed([normalized], batch_size=1)
            if query_vectors:
                vector_hits = self.vectors.top_k(query_vectors[0], limit=max(recall_k, top_k))
                for rank, (chunk_id, _score) in enumerate(vector_hits, start=1):
                    row = candidates.get(chunk_id) or self.lexical.get(chunk_id)
                    if row is not None and (not roles or row["role"] in roles):
                        candidates[chunk_id] = row
                        ranks.setdefault(chunk_id, {})["vector"] = rank
                        candidate_channels.setdefault(chunk_id, set()).add("vector")

        scored: list[tuple[dict, float, list[str]]] = []
        for chunk_id, row in candidates.items():
            channel_ranks = ranks.get(chunk_id, {})
            score = sum(1.0 / (60 + rank) for rank in channel_ranks.values())
            score += sum(
                self._heading_term_bonus(row, term)
                for term in query_terms
            )
            score += sum(0.008 for term in query_terms if term in row.get("terms", []))
            score *= 1.0 + max(int(row.get("authority", 0)), 0) / 2000.0
            scored.append((row, score, sorted(candidate_channels.get(chunk_id, set()))))

        if self.reranker and scored:
            rerank_scores = self.reranker.score(normalized, [row["retrieval_text"] for row, _, _ in scored])
            if len(rerank_scores) != len(scored):
                raise ValueError("Reranker returned a different number of scores than candidates")
            scored = [
                (row, score + float(rerank_score), channels + ["reranker"])
                for (row, score, channels), rerank_score in zip(scored, rerank_scores)
            ]

        scored.sort(key=lambda item: (item[1], int(item[0]["authority"])), reverse=True)
        deduped = self._dedupe_by_content(scored)
        selected = deduped[:top_k]
        if query_terms and (roles is None or PRIMARY_ROLE in roles):
            selected = self._ensure_primary_result(selected, deduped, top_k)
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
    def _heading_term_bonus(row: dict, term: str) -> float:
        """Favor the closest heading and avoid broad ancestor-title leakage."""

        if term in str(row.get("title", "")):
            return 0.03
        headings = [str(value) for value in row.get("heading_path", [])]
        for distance, heading in enumerate(reversed(headings)):
            if term in heading:
                return 0.024 / (distance + 1)
        return 0.0

    @staticmethod
    def _dedupe_by_content(scored: list[tuple[dict, float, list[str]]]) -> list[tuple[dict, float, list[str]]]:
        canonical: dict[str, tuple[dict, float, list[str]]] = {}
        for item in scored:
            digest = str(item[0].get("content_sha256", "")) or item[0]["id"]
            current = canonical.get(digest)
            if current is None or (
                int(item[0].get("authority", 0)), item[1]
            ) > (
                int(current[0].get("authority", 0)), current[1]
            ):
                canonical[digest] = item
        return sorted(
            canonical.values(),
            key=lambda item: (item[1], int(item[0].get("authority", 0))),
            reverse=True,
        )

    @staticmethod
    def _ensure_primary_result(
        selected: list[tuple[dict, float, list[str]]],
        candidates: list[tuple[dict, float, list[str]]],
        top_k: int,
    ) -> list[tuple[dict, float, list[str]]]:
        if not selected or any(item[0].get("role") == PRIMARY_ROLE for item in selected):
            return selected
        primary = next(
            (item for item in candidates if item[0].get("role") == PRIMARY_ROLE),
            None,
        )
        if primary is None or primary[1] < selected[-1][1] * PRIMARY_RESULT_SCORE_RATIO:
            return selected
        replacement = [*selected[: max(top_k - 1, 0)], primary]
        return sorted(
            replacement,
            key=lambda item: (item[1], int(item[0].get("authority", 0))),
            reverse=True,
        )

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
        indexed_terms = [str(term) for term in row["terms"]]
        taxonomy_terms = {
            group
            for term in indexed_terms
            for group in self.term_groups.get(term, [])
        }
        return RetrievalResult(
            chunk_id=row["id"],
            title=row["title"],
            text=row["text"],
            context_text="\n\n".join(item["text"] for item in context_rows),
            context_chunk_ids=[item["id"] for item in context_rows],
            source_path=row["source_path"],
            source_locators=row["source_locators"],
            heading_path=row["heading_path"],
            # Taxonomy labels come from the immutable terminology inventory,
            # not generated text. Exposing them here preserves the factual
            # parent category of a retrieved mechanism without editing either
            # the corpus or the persisted index.
            terms=[*indexed_terms, *sorted(taxonomy_terms - set(indexed_terms))],
            role=row["role"],
            authority=int(row["authority"]),
            score=round(score, 8),
            channels=channels,
        )
