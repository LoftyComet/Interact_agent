"""Persistent lexical and vector indexes for normalized knowledge chunks."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np


class Embedder(Protocol):
    embedding_model: str

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]: ...


def load_chunk_rows(chunks_path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(chunks_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunk_terms(path: str | Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["chunk_id"])] = [str(value) for value in row.get("terms", [])]
    return result


class LexicalIndex:
    """SQLite metadata store plus FTS5 BM25 search."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def build(cls, path: str | Path, chunks: list[dict], chunk_terms: dict[str, list[str]]) -> "LexicalIndex":
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        connection = sqlite3.connect(output)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    text TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_locators TEXT NOT NULL,
                    role TEXT NOT NULL,
                    authority INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    terms TEXT NOT NULL,
                    previous_chunk_id TEXT NOT NULL,
                    next_chunk_id TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    terms,
                    body,
                    tokenize='unicode61'
                );
                """
            )
            for chunk in chunks:
                terms = chunk_terms.get(chunk["id"], [])
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk["id"],
                        chunk["document_id"],
                        chunk["file_id"],
                        chunk["title"],
                        json.dumps(chunk.get("heading_path", []), ensure_ascii=False),
                        chunk["text"],
                        chunk["retrieval_text"],
                        chunk["source_path"],
                        json.dumps(chunk.get("source_locators", []), ensure_ascii=False),
                        chunk["role"],
                        int(chunk["authority"]),
                        chunk["content_sha256"],
                        json.dumps(terms, ensure_ascii=False),
                        chunk.get("previous_chunk_id", ""),
                        chunk.get("next_chunk_id", ""),
                    ),
                )
                connection.execute(
                    "INSERT INTO chunk_fts(chunk_id, title, terms, body) VALUES (?, ?, ?, ?)",
                    (
                        chunk["id"],
                        _lexicalize(" ".join([chunk["title"], *chunk.get("heading_path", [])])),
                        _lexicalize(" ".join(terms)),
                        _lexicalize(chunk["text"]),
                    ),
                )
            connection.execute("CREATE INDEX idx_chunks_document ON chunks(document_id)")
            connection.execute("CREATE INDEX idx_chunks_role ON chunks(role)")
            connection.execute("CREATE INDEX idx_chunks_hash ON chunks(content_sha256)")
            connection.commit()
        finally:
            connection.close()
        return cls(output)

    def search(self, query: str, limit: int = 30, roles: Optional[list[str]] = None) -> list[dict]:
        tokens = _lexical_tokens(query)
        if not tokens or limit <= 0:
            return []
        match_query = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        role_sql = ""
        params: list[Any] = [match_query]
        if roles:
            role_sql = f" AND c.role IN ({','.join('?' for _ in roles)})"
            params.extend(roles)
        params.append(limit)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                f"""
                SELECT c.*, -bm25(chunk_fts, 0.0, 8.0, 12.0, 1.0) AS score
                FROM chunk_fts
                JOIN chunks c ON c.id = chunk_fts.chunk_id
                WHERE chunk_fts MATCH ? {role_sql}
                ORDER BY score DESC, c.authority DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        finally:
            connection.close()
        return [_decode_chunk_row(dict(row)) for row in rows]

    def get(self, chunk_id: str) -> Optional[dict]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        finally:
            connection.close()
        return _decode_chunk_row(dict(row)) if row else None


class ChunkVectorIndex:
    """L2-normalized chunk embeddings with content-hash incremental reuse."""

    VECTORS_FILE = "vectors.npy"
    ROWS_FILE = "vector_rows.jsonl"
    META_FILE = "vector_meta.json"

    def __init__(self, vectors: np.ndarray, rows: list[dict], model: str):
        if vectors.shape[0] != len(rows):
            raise ValueError("Vector rows and metadata rows must have the same length")
        self.vectors = vectors.astype(np.float32)
        self.rows = rows
        self.model = model

    @classmethod
    def build(
        cls,
        chunks: list[dict],
        embedder: Embedder,
        existing: Optional["ChunkVectorIndex"] = None,
        *,
        batch_size: int = 32,
    ) -> "ChunkVectorIndex":
        cached: dict[str, np.ndarray] = {}
        if existing is not None and existing.model == embedder.embedding_model:
            cached = {
                row["content_sha256"]: existing.vectors[index]
                for index, row in enumerate(existing.rows)
            }
        missing = [chunk for chunk in chunks if chunk["content_sha256"] not in cached]
        if missing:
            generated = embedder.embed([chunk["retrieval_text"] for chunk in missing], batch_size=batch_size)
            matrix = _normalize(np.asarray(generated, dtype=np.float32))
            if matrix.shape[0] != len(missing):
                raise ValueError("Embedder returned a different number of vectors than requested")
            for chunk, vector in zip(missing, matrix):
                cached[chunk["content_sha256"]] = vector
        rows = [{"chunk_id": chunk["id"], "content_sha256": chunk["content_sha256"]} for chunk in chunks]
        if not rows:
            return cls(np.zeros((0, 0), dtype=np.float32), [], embedder.embedding_model)
        vectors = np.vstack([cached[row["content_sha256"]] for row in rows])
        return cls(vectors, rows, embedder.embedding_model)

    @classmethod
    def load(cls, directory: str | Path) -> "ChunkVectorIndex":
        root = Path(directory)
        vectors = np.load(root / cls.VECTORS_FILE)
        rows = [json.loads(line) for line in (root / cls.ROWS_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]
        meta = json.loads((root / cls.META_FILE).read_text(encoding="utf-8"))
        return cls(vectors, rows, str(meta["model"]))

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / self.VECTORS_FILE, self.vectors)
        with (root / self.ROWS_FILE).open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (root / self.META_FILE).write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "model": self.model,
                    "dimension": int(self.vectors.shape[1]) if self.vectors.size else 0,
                    "count": len(self.rows),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def top_k(self, query_vector: list[float] | np.ndarray, limit: int = 30) -> list[tuple[str, float]]:
        if not self.rows or limit <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query vector dimension does not match the index")
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm
        scores = self.vectors @ query
        limit = min(limit, len(self.rows))
        indexes = np.argpartition(-scores, limit - 1)[:limit]
        indexes = indexes[np.argsort(-scores[indexes])]
        return [(self.rows[index]["chunk_id"], float(scores[index])) for index in indexes]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _lexicalize(text: str) -> str:
    return " ".join(_lexical_tokens(text))


def _lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for run in re.findall(r"[\u3400-\u9fff]+|[A-Za-z0-9_+-]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", run):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        else:
            tokens.append(run)
    return list(dict.fromkeys(tokens))


def _decode_chunk_row(row: dict) -> dict:
    for field in ("heading_path", "source_locators", "terms"):
        row[field] = json.loads(row[field])
    return row
