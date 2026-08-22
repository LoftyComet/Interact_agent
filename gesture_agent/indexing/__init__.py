"""Offline corpus preparation and index-building primitives."""

from .extractors import ExtractionError, extract_document
from .chunking import chunk_documents, load_documents, write_chunks
from .inventory import build_inventory, load_inventory, write_inventory
from .indexes import ChunkVectorIndex, LexicalIndex, load_chunk_rows, load_chunk_terms
from .manifest import build_manifest, verify_manifest, write_manifest
from .models import CorpusFile, CorpusInventory, DocumentBlock, KnowledgeChunk, NormalizedDocument
from .pipeline import IndexBuildConfig, IndexBuildReport, build_knowledge_index
from .prepare import extract_inventory, write_build_report, write_documents
from .taxonomy import build_taxonomy, match_chunk_terms

__all__ = [
    "CorpusFile",
    "CorpusInventory",
    "ChunkVectorIndex",
    "DocumentBlock",
    "ExtractionError",
    "KnowledgeChunk",
    "IndexBuildConfig",
    "IndexBuildReport",
    "LexicalIndex",
    "NormalizedDocument",
    "build_inventory",
    "build_knowledge_index",
    "build_manifest",
    "build_taxonomy",
    "chunk_documents",
    "extract_document",
    "extract_inventory",
    "load_documents",
    "load_chunk_rows",
    "load_chunk_terms",
    "load_inventory",
    "match_chunk_terms",
    "verify_manifest",
    "write_build_report",
    "write_chunks",
    "write_documents",
    "write_inventory",
    "write_manifest",
]
