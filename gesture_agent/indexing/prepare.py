"""Orchestrate normalized-document extraction from a corpus inventory."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .extractors import ExtractionError, extract_document
from .models import CorpusInventory, NormalizedDocument


def extract_inventory(inventory: CorpusInventory) -> tuple[list[NormalizedDocument], dict]:
    """Extract all included files and return documents plus a build report."""

    root = Path(inventory.corpus_root)
    documents: list[NormalizedDocument] = []
    errors: list[dict[str, str]] = []
    for source in inventory.files:
        if source.status != "include":
            continue
        try:
            documents.append(extract_document(root / source.relative_path, source))
        except ExtractionError as exc:
            errors.append({"file_id": source.id, "source_path": source.relative_path, "error": str(exc)})
    report = {
        "schema_version": "1.0",
        "requested_count": inventory.summary.get("included_count", 0),
        "document_count": len(documents),
        "failed_count": len(errors),
        "block_count": sum(len(document.blocks) for document in documents),
        "errors": errors,
    }
    return documents, report


def write_documents(documents: list[NormalizedDocument], path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
    return output


def write_build_report(report: dict, path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
