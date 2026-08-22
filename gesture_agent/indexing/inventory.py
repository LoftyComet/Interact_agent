"""Discover, classify, hash, and de-duplicate local corpus files."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import CorpusFile, CorpusInventory, CorpusRole


SCHEMA_VERSION = "1.0"
SUPPORTED_EXTENSIONS = {".doc", ".docx", ".md", ".pdf", ".rtf", ".txt", ".xls", ".xlsx"}
IGNORED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "node_modules", "knowledge_index"}
PRIMARY_MARKER = "《交互方式通用手册》（词典2.0）"
SECONDARY_MARKER = "其他背景参考资料"


def build_inventory(root: str | Path) -> CorpusInventory:
    """Build an inventory without mutating the corpus.

    Exact duplicate files are detected by SHA-256. Classification is deliberately
    conservative: generated project data and annotations remain visible in the
    inventory, but are not eligible for semantic indexing.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Corpus root does not exist or is not a directory: {root_path}")

    discovered: list[CorpusFile] = []
    for path in _iter_candidates(root_path):
        relative = path.relative_to(root_path).as_posix()
        digest = _sha256(path)
        role, authority, status, reason = _classify(relative, path.name)
        stat = path.stat()
        discovered.append(
            CorpusFile(
                id=_file_id(relative),
                relative_path=relative,
                extension=path.suffix.lower(),
                media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                sha256=digest,
                role=role,
                authority=authority,
                status=status,
                reason=reason,
                duplicate_of="",
            )
        )

    files = _mark_exact_duplicates(discovered)
    summary = _summarize(files)
    return CorpusInventory(
        schema_version=SCHEMA_VERSION,
        corpus_root=str(root_path),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        files=files,
        summary=summary,
    )


def write_inventory(inventory: CorpusInventory, output_path: str | Path) -> Path:
    """Write an inventory as UTF-8 JSON and return the resolved output path."""

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_inventory(path: str | Path) -> CorpusInventory:
    """Load an inventory previously produced by :func:`write_inventory`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    files = [CorpusFile(**item) for item in payload.get("files", [])]
    return CorpusInventory(
        schema_version=str(payload["schema_version"]),
        corpus_root=str(payload["corpus_root"]),
        generated_at=str(payload["generated_at"]),
        files=files,
        summary=dict(payload.get("summary", {})),
    )


def _iter_candidates(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in IGNORED_PARTS for part in relative_parts):
            continue
        yield path


def _classify(relative_path: str, filename: str) -> tuple[CorpusRole, int, str, str]:
    parts = Path(relative_path).parts
    lowered = relative_path.lower()

    if filename.startswith(".~") or filename.startswith("~$"):
        return "unknown", 0, "exclude", "temporary_office_file"
    if any(part.startswith(".") for part in parts):
        return "derived", 0, "exclude", "hidden_generated_content"
    if relative_path.startswith("Interact_agent/"):
        if "/data/" in f"/{relative_path}" or "/docs/" in f"/{relative_path}":
            return "derived", 20, "exclude", "project_generated_or_reference_copy"
        return "derived", 0, "exclude", "project_file"
    if filename == "用户输入.xlsx":
        return "evaluation", 0, "exclude", "evaluation_dataset"
    if filename.endswith("标注.xlsx") or "大纲" in filename:
        return "annotation", 0, "exclude", "annotation_or_taxonomy"
    if PRIMARY_MARKER in relative_path:
        if filename.startswith("（合）") or filename.startswith("(合)"):
            return "primary", 100, "exclude", "aggregate_copy_of_chapter_files"
        return "primary", 100, "include", "canonical_handbook_chapter"
    if SECONDARY_MARKER in relative_path:
        return "secondary", 60, "include", "background_reference"
    if lowered.endswith((".md", ".doc", ".docx", ".pdf", ".rtf", ".txt")):
        return "unknown", 40, "include", "unclassified_textual_source"
    return "unknown", 0, "exclude", "non_textual_or_unclassified"


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _file_id(relative_path: str) -> str:
    return "file_" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def _summarize(files: list[CorpusFile]) -> dict:
    included = [item for item in files if item.status == "include"]
    duplicate_count = sum(bool(item.duplicate_of) for item in files)
    return {
        "file_count": len(files),
        "included_count": len(included),
        "excluded_count": len(files) - len(included),
        "duplicate_count": duplicate_count,
        "included_bytes": sum(item.size_bytes for item in included),
        "by_extension": dict(sorted(Counter(item.extension for item in files).items())),
        "by_role": dict(sorted(Counter(item.role for item in files).items())),
        "by_status": dict(sorted(Counter(item.status for item in files).items())),
    }


def _mark_exact_duplicates(files: list[CorpusFile]) -> list[CorpusFile]:
    """Keep the highest-authority copy as canonical, independent of scan order."""

    grouped: dict[str, list[CorpusFile]] = {}
    for item in files:
        grouped.setdefault(item.sha256, []).append(item)

    result: list[CorpusFile] = []
    for group in grouped.values():
        canonical = sorted(
            group,
            key=lambda item: (
                item.status != "include",
                -item.authority,
                item.relative_path,
            ),
        )[0]
        result.append(canonical)
        for item in group:
            if item.id == canonical.id:
                continue
            result.append(
                replace(
                    item,
                    status="exclude",
                    reason=f"exact_duplicate:{canonical.id}",
                    duplicate_of=canonical.id,
                )
            )
    return sorted(result, key=lambda item: item.relative_path)
