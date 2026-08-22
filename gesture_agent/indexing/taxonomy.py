"""Compile expert terminology and structured knowledge into portable indexes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_taxonomy(term_inventory_path: str | Path, structured_knowledge_path: str | Path) -> tuple[dict, list[dict]]:
    """Return a normalized terminology payload and de-duplicated relation rows."""

    inventory = json.loads(Path(term_inventory_path).read_text(encoding="utf-8"))
    structured = json.loads(Path(structured_knowledge_path).read_text(encoding="utf-8"))
    terms: list[dict] = []
    aliases: dict[str, str] = {}
    relations: list[dict] = []

    root = inventory.get("root")
    if isinstance(root, dict):
        _walk_tree(root, [], None, None, terms, aliases, relations)
    else:
        for layer, labels in inventory.get("by_layer", {}).items():
            for label in labels:
                terms.append(_term_record(str(label), layer=layer))
        _merge_alias_config(inventory.get("aliases", {}), aliases)

    known = {item["label"] for item in terms}
    raw_items = structured.get("items", structured) if isinstance(structured, dict) else structured
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict) or not item.get("term"):
                continue
            label = str(item["term"]).strip()
            if label not in known:
                terms.append(
                    _term_record(
                        label,
                        label_en="",
                        layer=str(item.get("layer", "unknown")),
                        term_type=str(item.get("term_type", "")),
                        aliases=[str(value) for value in item.get("aliases", [])],
                        source="structured_knowledge",
                    )
                )
                known.add(label)
            for alias in item.get("aliases", []):
                if str(alias).strip():
                    aliases[str(alias).strip()] = label
            for field, relation_type in (
                ("properties", "has_property"),
                ("mechanisms", "uses_mechanism"),
                ("control_forms", "uses_control_form"),
                ("related_terms", "related_to"),
            ):
                for target in item.get(field, []):
                    if str(target).strip():
                        relations.append(_relation(label, relation_type, str(target).strip(), "structured_knowledge"))

    _merge_alias_config(inventory.get("aliases", {}), aliases)
    for alias, canonical in aliases.items():
        relations.append(_relation(alias, "alias_of", canonical, "term_inventory"))
    terms = sorted(_dedupe_terms(terms), key=lambda item: (item.get("layer", ""), item["label"]))
    relations = _dedupe_relations(relations)
    payload = {
        "schema_version": "1.0",
        "source": inventory.get("source", str(term_inventory_path)),
        "structural_terms": list(inventory.get("structural_terms", [])),
        "aliases": dict(sorted(aliases.items())),
        "terms": terms,
        "summary": {
            "term_count": len(terms),
            "alias_count": len(aliases),
            "relation_count": len(relations),
        },
    }
    return payload, relations


def match_chunk_terms(chunks_path: str | Path, terminology: dict) -> list[dict]:
    """Create a compact chunk-to-canonical-term lookup for retrieval boosts."""

    canonical = [str(item["label"]) for item in terminology.get("terms", [])]
    aliases = {str(key): str(value) for key, value in terminology.get("aliases", {}).items()}
    candidates = sorted({*canonical, *aliases}, key=lambda value: (-len(value), value))
    rows: list[dict] = []
    for raw_line in Path(chunks_path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        chunk = json.loads(raw_line)
        haystack = f"{' '.join(chunk.get('heading_path', []))}\n{chunk.get('text', '')}".lower()
        found: list[str] = []
        for candidate in candidates:
            if candidate.lower() in haystack:
                normalized = aliases.get(candidate, candidate)
                if normalized not in found:
                    found.append(normalized)
        rows.append({"chunk_id": chunk["id"], "terms": found})
    return rows


def write_json(payload: dict, path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def write_jsonl(rows: list[dict], path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def _walk_tree(
    node: dict,
    path: list[str],
    inherited_layer: str | None,
    inherited_type: str | None,
    terms: list[dict],
    aliases: dict[str, str],
    relations: list[dict],
) -> None:
    label = str(node.get("label", "")).strip()
    layer = str(node.get("layer") or inherited_layer or "unknown")
    term_type = str(node.get("type") or inherited_type or "")
    children = node.get("children") if isinstance(node.get("children"), list) else []
    next_path = [*path, label] if label else list(path)
    if label and not children:
        node_aliases = [str(value).strip() for value in node.get("aliases", []) if str(value).strip()]
        terms.append(
            _term_record(
                label,
                label_en=str(node.get("label_en", "")),
                code=str(node.get("code", "")),
                description=str(node.get("desc", "")),
                layer=layer,
                term_type=term_type,
                group_path=path,
                aliases=node_aliases,
                source="term_inventory",
            )
        )
        for alias in node_aliases:
            aliases[alias] = label
        for group in path:
            relations.append(_relation(label, "member_of", group, "term_inventory"))
    for child in children:
        if isinstance(child, dict):
            _walk_tree(child, next_path, layer, term_type, terms, aliases, relations)


def _term_record(
    label: str,
    *,
    label_en: str = "",
    code: str = "",
    description: str = "",
    layer: str = "unknown",
    term_type: str = "",
    group_path: list[str] | None = None,
    aliases: list[str] | None = None,
    source: str = "term_inventory",
) -> dict:
    return {
        "id": "term_" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:16],
        "label": label,
        "label_en": label_en,
        "code": code,
        "description": description,
        "layer": layer,
        "term_type": term_type,
        "group_path": group_path or [],
        "aliases": aliases or [],
        "source": source,
    }


def _relation(source: str, relation_type: str, target: str, evidence_source: str) -> dict:
    identity = f"{source}\0{relation_type}\0{target}"
    return {
        "id": "rel_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "source": source,
        "type": relation_type,
        "target": target,
        "evidence_source": evidence_source,
    }


def _merge_alias_config(raw: Any, aliases: dict[str, str]) -> None:
    if not isinstance(raw, dict):
        return
    for key, value in raw.items():
        if isinstance(value, dict):
            _merge_alias_config(value, aliases)
        elif str(key).strip() and str(value).strip():
            aliases[str(key).strip()] = str(value).strip()


def _dedupe_terms(terms: list[dict]) -> list[dict]:
    by_label: dict[str, dict] = {}
    for term in terms:
        by_label.setdefault(term["label"], term)
    return list(by_label.values())


def _dedupe_relations(relations: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in relations}
    return sorted(by_id.values(), key=lambda row: (row["source"], row["type"], row["target"]))
