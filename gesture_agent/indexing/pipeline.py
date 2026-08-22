from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chunking import chunk_documents, write_chunks
from .indexes import ChunkVectorIndex, Embedder, LexicalIndex
from .inventory import build_inventory, write_inventory
from .manifest import build_manifest, write_manifest
from .prepare import extract_inventory, write_build_report, write_documents
from .taxonomy import build_taxonomy, match_chunk_terms, write_json, write_jsonl


@dataclass(frozen=True)
class IndexBuildConfig:
    corpus_root: Path
    output_dir: Path
    term_inventory_path: Path
    structured_knowledge_path: Path
    max_chars: int = 1000
    overlap_chars: int = 120
    vector_batch_size: int = 32


@dataclass(frozen=True)
class IndexBuildReport:
    index_dir: str
    build_id: str
    document_count: int
    chunk_count: int
    term_count: int
    extraction_failures: int
    embedding_status: str


def build_knowledge_index(
    config: IndexBuildConfig,
    *,
    embedder: Optional[Embedder] = None,
) -> IndexBuildReport:
    """Build every portable index artifact behind one stable entry point."""

    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(config.corpus_root)
    write_inventory(inventory, output / "corpus_inventory.json")

    documents, extraction_report = extract_inventory(inventory)
    write_documents(documents, output / "documents.jsonl")
    write_build_report(extraction_report, output / "extraction_report.json")
    if not documents:
        raise RuntimeError("Corpus extraction produced no documents; inspect corpus_inventory.json")

    chunks, chunk_report = chunk_documents(
        documents,
        max_chars=config.max_chars,
        overlap_chars=config.overlap_chars,
    )
    write_chunks(chunks, output / "chunks.jsonl")
    write_build_report(chunk_report, output / "chunk_report.json")

    terminology, relations = build_taxonomy(
        config.term_inventory_path,
        config.structured_knowledge_path,
    )
    chunk_terms = match_chunk_terms(output / "chunks.jsonl", terminology)
    write_json(terminology, output / "terminology.json")
    write_jsonl(relations, output / "relations.jsonl")
    write_jsonl(chunk_terms, output / "chunk_terms.jsonl")

    chunk_rows = [chunk.to_dict() for chunk in chunks]
    terms_by_chunk = {row["chunk_id"]: row["terms"] for row in chunk_terms}
    LexicalIndex.build(output / "knowledge.sqlite", chunk_rows, terms_by_chunk)

    if embedder is not None:
        vector_dir = output / "vectors"
        existing = None
        if (vector_dir / ChunkVectorIndex.META_FILE).is_file():
            existing = ChunkVectorIndex.load(vector_dir)
        vectors = ChunkVectorIndex.build(
            chunk_rows,
            embedder,
            existing=existing,
            batch_size=config.vector_batch_size,
        )
        vectors.save(vector_dir)

    manifest = build_manifest(output)
    write_manifest(manifest, output / "manifest.json")
    return IndexBuildReport(
        index_dir=str(output),
        build_id=str(manifest["build_id"]),
        document_count=len(documents),
        chunk_count=len(chunks),
        term_count=int(terminology["summary"]["term_count"]),
        extraction_failures=int(extraction_report["failed_count"]),
        embedding_status=str(manifest["embedding"]["status"]),
    )
