from __future__ import annotations

import json

from gesture_agent.indexing import LexicalIndex, build_manifest, write_manifest
from gesture_agent.knowledge import KnowledgeBase


def test_knowledge_base_can_switch_to_deployable_index(tmp_path) -> None:
    chunk = {
        "id": "chunk_knob",
        "document_id": "doc_knob",
        "file_id": "file_knob",
        "title": "控件定义",
        "heading_path": ["控件形态", "旋钮", "控件定义"],
        "text": "旋钮可承载角度属性并用于连续调节。",
        "retrieval_text": "控件形态 旋钮 控件定义\n旋钮可承载角度属性并用于连续调节。",
        "source_path": "手册/控件形态.docx",
        "source_locators": [{"paragraph": 12}],
        "role": "primary",
        "authority": 100,
        "content_sha256": "knob-hash",
        "previous_chunk_id": "",
        "next_chunk_id": "",
    }
    LexicalIndex.build(tmp_path / "knowledge.sqlite", [chunk], {"chunk_knob": ["旋钮", "角度属性"]})
    (tmp_path / "corpus_inventory.json").write_text(json.dumps({"corpus_root": "/corpus"}), encoding="utf-8")
    (tmp_path / "chunk_report.json").write_text(json.dumps({"document_count": 1, "chunk_count": 1}), encoding="utf-8")
    (tmp_path / "terminology.json").write_text(
        json.dumps({"aliases": {}, "terms": [{"label": "旋钮"}], "summary": {"term_count": 1}}, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = build_manifest(tmp_path)
    write_manifest(manifest, tmp_path / "manifest.json")

    kb = KnowledgeBase.load("data", index_dir=tmp_path)
    results = kb.search("旋钮有哪些属性", top_k=1, prefer_terms=["旋钮"])

    assert kb.retriever is not None
    assert kb.index_chunk_count == 1
    assert results[0].id == "chunk_knob"
    assert results[0].source == "手册/控件形态.docx"
    assert results[0].start_line == 13
    assert results[0].layer == "control_form"
