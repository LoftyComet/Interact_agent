from __future__ import annotations

import json

from gesture_agent.indexing.manifest import build_manifest, verify_manifest, write_manifest


def test_manifest_hashes_and_verifies_artifacts(tmp_path) -> None:
    (tmp_path / "corpus_inventory.json").write_text(json.dumps({"corpus_root": "/corpus"}), encoding="utf-8")
    (tmp_path / "chunk_report.json").write_text(json.dumps({"document_count": 2, "chunk_count": 8}), encoding="utf-8")
    (tmp_path / "terminology.json").write_text(json.dumps({"summary": {"term_count": 4}}), encoding="utf-8")
    (tmp_path / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    manifest = build_manifest(tmp_path)
    write_manifest(manifest, tmp_path / "manifest.json")

    assert manifest["chunk_count"] == 8
    assert manifest["embedding"]["status"] == "not_built"
    assert verify_manifest(tmp_path, manifest) == []
    (tmp_path / "chunks.jsonl").write_text("changed", encoding="utf-8")
    assert verify_manifest(tmp_path, manifest) == ["size_mismatch:chunks.jsonl"]
