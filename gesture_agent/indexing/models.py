"""Stable data models shared by offline indexing modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


CorpusRole = Literal["primary", "secondary", "annotation", "evaluation", "derived", "unknown"]
CorpusStatus = Literal["include", "exclude"]
BlockKind = Literal["heading", "paragraph", "table"]


@dataclass(frozen=True)
class CorpusFile:
    """One file discovered beneath a corpus root."""

    id: str
    relative_path: str
    extension: str
    media_type: str
    size_bytes: int
    modified_at: str
    sha256: str
    role: CorpusRole
    authority: int
    status: CorpusStatus
    reason: str = ""
    duplicate_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusInventory:
    """Versioned, portable description of corpus inputs."""

    schema_version: str
    corpus_root: str
    generated_at: str
    files: list[CorpusFile] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_root": self.corpus_root,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True)
class DocumentBlock:
    """A source-locatable block extracted from a document."""

    id: str
    sequence: int
    kind: BlockKind
    text: str
    heading_level: int | None = None
    source_locator: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedDocument:
    """Format-neutral document representation consumed by the chunker."""

    schema_version: str
    id: str
    file_id: str
    title: str
    source_path: str
    source_sha256: str
    source_format: str
    role: CorpusRole
    authority: int
    blocks: list[DocumentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in asdict(self).items() if key != "blocks"},
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class KnowledgeChunk:
    """Retrieval unit with stable identity and source lineage."""

    schema_version: str
    id: str
    document_id: str
    file_id: str
    title: str
    heading_path: list[str]
    text: str
    retrieval_text: str
    source_path: str
    source_locators: list[dict[str, Any]]
    role: CorpusRole
    authority: int
    content_sha256: str
    char_count: int
    previous_chunk_id: str = ""
    next_chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
