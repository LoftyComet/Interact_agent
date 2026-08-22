from __future__ import annotations

import json

from gesture_agent.indexing import IndexBuildConfig, build_knowledge_index, verify_manifest
from gesture_agent.retrieval import HybridRetriever


def test_pipeline_builds_a_loadable_lexical_bundle(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chapter.md").write_text(
        "# 旋钮\n\n## 控件定义\n旋钮可以承载角度属性并用于连续调节。\n",
        encoding="utf-8",
    )
    term_inventory = tmp_path / "terms.json"
    term_inventory.write_text(
        json.dumps({"by_layer": {"control_form": ["旋钮"], "basic_property": ["角度属性"]}}),
        encoding="utf-8",
    )
    structured = tmp_path / "structured.json"
    structured.write_text(json.dumps({"items": []}), encoding="utf-8")
    output = tmp_path / "index"

    report = build_knowledge_index(IndexBuildConfig(
        corpus_root=corpus,
        output_dir=output,
        term_inventory_path=term_inventory,
        structured_knowledge_path=structured,
        max_chars=300,
        overlap_chars=30,
    ))

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report.document_count == 1
    assert report.chunk_count == 1
    assert report.embedding_status == "not_built"
    assert verify_manifest(output, manifest) == []
    hits = HybridRetriever.from_directory(output).retrieve("旋钮角度", top_k=1)
    assert hits[0].title == "控件定义"
