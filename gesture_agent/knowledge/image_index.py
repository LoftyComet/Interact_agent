"""Image index for retrieving relevant illustrations from extracted docx images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gesture_agent.core.models import SourceChunk
from gesture_agent.core.text_utils import tokenize


@dataclass
class ImageEntry:
    id: str
    filename: str
    source_docx: str
    heading: str
    annotation: str
    terms: list[str] = field(default_factory=list)

    def all_text(self) -> str:
        return f"{self.heading} {self.annotation}"

    @property
    def reference_id(self) -> str:
        """Return a compact token that models can reproduce without truncation."""

        digest = hashlib.sha256(self.id.encode("utf-8")).hexdigest()[:12]
        return f"img_{digest}"


class ImageIndex:
    def __init__(self, entries: list[ImageEntry], base_dir: Path) -> None:
        self.entries = entries
        self.base_dir = base_dir

    @classmethod
    def load(cls, index_path: Path) -> Optional["ImageIndex"]:
        if not index_path.exists():
            return None
        data = json.loads(index_path.read_text(encoding="utf-8"))
        entries = []
        for item in data.get("images", []):
            entries.append(ImageEntry(
                id=item["id"],
                filename=item["filename"],
                source_docx=item.get("source_docx", ""),
                heading=item.get("heading", ""),
                annotation=item.get("annotation", ""),
                terms=item.get("terms", []),
            ))
        return cls(entries, index_path.parent)

    def get_filename(self, image_id: str) -> Optional[str]:
        for entry in self.entries:
            if entry.id == image_id or entry.reference_id == image_id:
                return entry.filename
        return None

    def search(
        self,
        query_terms: list[str],
        chunks: list[SourceChunk],
        top_k: int = 3,
    ) -> list[ImageEntry]:
        if not self.entries:
            return []

        chunk_terms: set[str] = set()
        chunk_titles: set[str] = set()
        for c in chunks:
            chunk_terms.update(c.terms)
            chunk_titles.add(c.title)

        scored: list[tuple[float, ImageEntry]] = []
        for entry in self.entries:
            score = 0.0
            entry_tokens = set(tokenize(entry.all_text()))
            for term in query_terms:
                if term in entry.terms:
                    score += 5.0
                elif term in entry.heading or term in entry.annotation:
                    score += 3.0
                elif term in entry_tokens:
                    score += 1.0
            for term in chunk_terms:
                if term in entry.terms:
                    score += 3.0
                elif term in entry.heading:
                    score += 2.0
            for title in chunk_titles:
                if title in entry.heading or entry.heading in title:
                    score += 4.0
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [entry for _, entry in scored[:top_k]]
