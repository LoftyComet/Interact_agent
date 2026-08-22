from __future__ import annotations

import json

from gesture_agent.indexing.taxonomy import build_taxonomy, match_chunk_terms


def test_taxonomy_flattens_tree_aliases_and_relations(tmp_path) -> None:
    inventory = tmp_path / "terms.json"
    inventory.write_text(
        json.dumps(
            {
                "schema": "tree",
                "root": {
                    "label": "基础概念",
                    "children": [
                        {
                            "label": "交互逻辑",
                            "layer": "interaction_mechanism",
                            "children": [
                                {"label": "单击", "label_en": "Single Tap", "aliases": ["点一下"], "code": "1-b"}
                            ],
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    structured = tmp_path / "structured.json"
    structured.write_text(
        json.dumps({"items": [{"term": "单击", "mechanisms": ["按下"], "related_terms": ["双击"]}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    terminology, relations = build_taxonomy(inventory, structured)

    assert terminology["terms"][0]["label"] == "单击"
    assert terminology["aliases"]["点一下"] == "单击"
    assert any(row["type"] == "uses_mechanism" and row["target"] == "按下" for row in relations)


def test_chunk_term_matching_normalizes_aliases(tmp_path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({"id": "c1", "heading_path": ["点击"], "text": "用户点一下按钮。"}, ensure_ascii=False) + "\n", encoding="utf-8")
    terminology = {"terms": [{"label": "单击"}], "aliases": {"点一下": "单击"}}

    rows = match_chunk_terms(chunks, terminology)

    assert rows == [{"chunk_id": "c1", "terms": ["单击"]}]
