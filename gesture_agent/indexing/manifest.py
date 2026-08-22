"""Version and integrity metadata for deployable knowledge-index bundles."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ARTIFACTS = (
    "corpus_inventory.json",
    "documents.jsonl",
    "extraction_report.json",
    "chunks.jsonl",
    "chunk_report.json",
    "terminology.json",
    "relations.jsonl",
    "chunk_terms.jsonl",
    "knowledge.sqlite",
    "vectors/vectors.npy",
    "vectors/vector_rows.jsonl",
    "vectors/vector_meta.json",
)


def build_manifest(index_dir: str | Path) -> dict:
    root = Path(index_dir).resolve()
    inventory = _load_json(root / "corpus_inventory.json")
    chunk_report = _load_json(root / "chunk_report.json")
    terminology = _load_json(root / "terminology.json")
    vector_meta = _load_json(root / "vectors" / "vector_meta.json", required=False)
    artifacts: list[dict] = []
    for relative in DEFAULT_ARTIFACTS:
        path = root / relative
        if path.is_file():
            artifacts.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "schema_version": "1.0",
        "build_id": "build_" + hashlib.sha256(
            "\n".join(f"{item['path']}:{item['sha256']}" for item in artifacts).encode("utf-8")
        ).hexdigest()[:16],
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "document_count": int(chunk_report.get("document_count", 0)),
        "chunk_count": int(chunk_report.get("chunk_count", 0)),
        "term_count": int(terminology.get("summary", {}).get("term_count", 0)),
        "corpus_root_at_build_time": inventory.get("corpus_root", ""),
        "embedding": {
            "status": "ready" if vector_meta else "not_built",
            "model": vector_meta.get("model", "") if vector_meta else "",
            "dimension": int(vector_meta.get("dimension", 0)) if vector_meta else 0,
            "count": int(vector_meta.get("count", 0)) if vector_meta else 0,
        },
        "artifacts": artifacts,
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def verify_manifest(index_dir: str | Path, manifest: dict) -> list[str]:
    root = Path(index_dir).resolve()
    errors: list[str] = []
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if not path.is_file():
            errors.append(f"missing:{artifact['path']}")
        elif path.stat().st_size != artifact["size_bytes"]:
            errors.append(f"size_mismatch:{artifact['path']}")
        elif _sha256(path) != artifact["sha256"]:
            errors.append(f"hash_mismatch:{artifact['path']}")
    return errors


def _load_json(path: Path, *, required: bool = True) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
