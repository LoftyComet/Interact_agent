from __future__ import annotations

import json

from gesture_agent.indexing.inventory import build_inventory, write_inventory


def test_inventory_classifies_sources_and_exact_duplicates(tmp_path) -> None:
    primary = tmp_path / "IxDL资料" / "《交互方式通用手册》（词典2.0）"
    primary.mkdir(parents=True)
    chapter = primary / "1 第一篇.docx"
    chapter.write_bytes(b"chapter-one")
    aggregate = primary / "（合）《交互方式通用手册》（词典2.0）.docx"
    aggregate.write_bytes(b"aggregate")
    annotation = tmp_path / "output_frames_标注.xlsx"
    annotation.write_bytes(b"annotation")
    duplicate = tmp_path / "copy.md"
    duplicate.write_bytes(b"chapter-one")

    inventory = build_inventory(tmp_path)
    by_path = {item.relative_path: item for item in inventory.files}

    assert by_path[chapter.relative_to(tmp_path).as_posix()].status == "include"
    assert by_path[chapter.relative_to(tmp_path).as_posix()].authority == 100
    assert by_path[aggregate.relative_to(tmp_path).as_posix()].reason == "aggregate_copy_of_chapter_files"
    assert by_path[annotation.name].role == "annotation"
    assert by_path[duplicate.name].duplicate_of == by_path[chapter.relative_to(tmp_path).as_posix()].id
    assert inventory.summary["duplicate_count"] == 1


def test_inventory_excludes_hidden_generated_and_office_temp_files(tmp_path) -> None:
    hidden = tmp_path / ".codex-skills" / "answer.md"
    hidden.parent.mkdir()
    hidden.write_text("generated", encoding="utf-8")
    temp = tmp_path / ".~draft.docx"
    temp.write_bytes(b"temp")

    inventory = build_inventory(tmp_path)
    by_path = {item.relative_path: item for item in inventory.files}

    assert by_path[".codex-skills/answer.md"].reason == "hidden_generated_content"
    assert by_path[".~draft.docx"].reason == "temporary_office_file"


def test_inventory_json_round_trip(tmp_path) -> None:
    source = tmp_path / "source.md"
    source.write_text("# 标题\n正文", encoding="utf-8")
    inventory = build_inventory(tmp_path)

    output = write_inventory(inventory, tmp_path / "out" / "inventory.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["file_count"] == 1
    assert payload["files"][0]["sha256"]
