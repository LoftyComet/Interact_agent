from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Union

from gesture_agent.core.models import Layer, SourceChunk, TermInventory
from gesture_agent.core.text_utils import clean_text, compact_whitespace, strip_heading_prefix, tokenize


MECHANISM_HEADING_RE = re.compile(r"^\d+-[a-z]\s+[^、，,]{1,80}$", re.IGNORECASE)
PROPERTY_HEADING_RE = re.compile(r"^[（(]\d+[）)][^,，。；;]{0,30}(?:属性|信号|阶次控制)[^,，。；;]{0,40}$")
FORM_HEADING_RE = re.compile(r"^[（(]\d+[）)][^：:]{1,24}$")
TOP_HEADING_RE = re.compile(r"^\d+\.\s+[^：:]{2,40}$")
COMPARISON_FILENAME = "交互机制对比.md"
COMPARISON_HEADING_RE = re.compile(r"^\d+、.{2,80}$")
COMPARISON_SPLIT_RE = re.compile(r"\s*(?:vs|VS|Vs|和|与|、|/)\s*")
TERM_CODE_RE = re.compile(r"^(?:\d+-[a-z]|[0-9A-Za-z /×÷+\-]+)$", re.IGNORECASE)
STRUCTURAL_TERMS = [
    "控件形态",
    "基本属性",
    "基础属性",
    "交互机制",
    "响应逻辑",
    "交互特性",
    "适用边界",
    "不适用场景",
    "系统反馈",
    "状态/变化序列",
    "方案复述",
    "结构拆解",
    "问题诊断",
    "修改建议",
    "规范术语版本",
]
DEFAULT_TERM_INVENTORY_FILENAME = "term_inventory.json"
TERM_INVENTORY_MODES = {"merge", "replace"}


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        term_inventory_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.term_inventory_path = Path(term_inventory_path) if term_inventory_path else self.data_dir / DEFAULT_TERM_INVENTORY_FILENAME
        self.chunks: list[SourceChunk] = []
        self.terms: list[str] = []
        self.term_inventory = TermInventory()

    @classmethod
    def load(
        cls,
        data_dir: Union[str, Path] = "data",
        term_inventory_path: Optional[Union[str, Path]] = None,
    ) -> "KnowledgeBase":
        kb = cls(data_dir, term_inventory_path=term_inventory_path)
        kb._load()
        return kb

    def _load(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        chunks: list[SourceChunk] = []
        for path in sorted(self.data_dir.glob("*.md")):
            chunks.extend(self._parse_markdown(path))

        self.chunks = chunks
        term_set: set[str] = set()
        for chunk in chunks:
            term_set.update(chunk.terms)
        self.terms = sorted(term_set, key=lambda item: (-len(item), item))
        generated_inventory = self._build_term_inventory(chunks)
        self.term_inventory = self._load_term_inventory_config(generated_inventory)
        term_set.update(self.term_inventory.all_terms())
        self.terms = sorted(term_set, key=lambda item: (-len(item), item))

    def _parse_markdown(self, path: Path) -> list[SourceChunk]:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        lines = [clean_text(line).rstrip() for line in raw_lines]
        heading_indexes = self._heading_indexes(path.name, lines)
        if not heading_indexes:
            whole_text = "\n".join(line for line in lines if line.strip())
            return [
                SourceChunk(
                    id=f"{path.name}:1",
                    title=path.stem,
                    source=str(path),
                    start_line=1,
                    end_line=len(lines),
                    text=whole_text,
                    layer="unknown",
                    terms=[path.stem],
                )
            ]

        parsed: list[SourceChunk] = []
        for pos, start_idx in enumerate(heading_indexes):
            end_idx = heading_indexes[pos + 1] if pos + 1 < len(heading_indexes) else len(lines)
            title = lines[start_idx].strip()
            body_lines = [line for line in lines[start_idx:end_idx] if line.strip()]
            if not body_lines:
                continue
            text = "\n".join(body_lines)
            layer = self._infer_layer(path.name, title)
            terms = self._extract_terms(title)
            parsed.append(
                SourceChunk(
                    id=f"{path.name}:{start_idx + 1}",
                    title=title,
                    source=str(path),
                    start_line=start_idx + 1,
                    end_line=end_idx,
                    text=text,
                    layer=layer,
                    terms=terms,
                )
            )
        return parsed

    def _heading_indexes(self, filename: str, lines: list[str]) -> list[int]:
        indexes: list[int] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if filename == COMPARISON_FILENAME:
                if idx == 0 or COMPARISON_HEADING_RE.match(stripped):
                    indexes.append(idx)
                continue
            if MECHANISM_HEADING_RE.match(stripped):
                indexes.append(idx)
                continue
            if filename == "1.md" and PROPERTY_HEADING_RE.match(stripped):
                indexes.append(idx)
                continue
            if filename == "3.md" and FORM_HEADING_RE.match(stripped):
                indexes.append(idx)
                continue
            if TOP_HEADING_RE.match(stripped) and filename in {"1.md", "2.md", "3.md"}:
                indexes.append(idx)
        return sorted(set(indexes))

    def _infer_layer(self, filename: str, title: str) -> Layer:
        if filename == COMPARISON_FILENAME:
            return "interaction_mechanism"
        if MECHANISM_HEADING_RE.match(title):
            return "interaction_mechanism"
        if "多模态" in title:
            return "multimodal_interaction"
        if "属性" in title or "生理信号" in title or "阶次控制" in title:
            return "basic_property"
        if filename == "3.md" and FORM_HEADING_RE.match(title):
            return "control_form"
        if "案例" in title:
            return "interaction_case"
        return "unknown"

    def _extract_terms(self, title: str) -> list[str]:
        cleaned = clean_text(title)
        terms: list[str] = []
        base = strip_heading_prefix(cleaned)
        if base:
            terms.append(base)
            if "vs" in base.lower() or "与" in base or "和" in base or "、" in base or "/" in base:
                terms.extend(
                    item.strip()
                    for item in COMPARISON_SPLIT_RE.split(base)
                    if 1 < len(item.strip()) <= 24
                )
        code_match = re.match(r"^(\d+-[a-z])\s+(.+)$", cleaned, flags=re.IGNORECASE)
        if code_match:
            terms.append(code_match.group(1).lower())
            name = strip_heading_prefix(code_match.group(2))
            if name:
                terms.append(name)
        cn_paren = re.findall(r"[（(]([^）)]+)[）)]", cleaned)
        for item in cn_paren:
            item = item.strip()
            if item and len(item) <= 24 and not re.fullmatch(r"[0-9A-Za-z /×÷+-]+", item):
                terms.append(item)
        unique: list[str] = []
        seen: set[str] = set()
        for term in terms:
            term = compact_whitespace(term)
            if not term or term in seen:
                continue
            seen.add(term)
            unique.append(term)
        return unique

    def _build_term_inventory(self, chunks: list[SourceChunk]) -> TermInventory:
        by_layer: dict[Layer, list[str]] = {}
        for chunk in chunks:
            layer_terms = by_layer.setdefault(chunk.layer, [])
            for term in chunk.terms:
                canonical = _canonical_output_term(term)
                if canonical and canonical not in layer_terms:
                    layer_terms.append(canonical)

        for layer, terms in list(by_layer.items()):
            by_layer[layer] = sorted(terms, key=lambda item: (-len(item), item))
        return TermInventory(by_layer=by_layer, structural_terms=STRUCTURAL_TERMS, source="generated")

    def _load_term_inventory_config(self, generated: TermInventory) -> TermInventory:
        if not self.term_inventory_path.exists():
            return generated

        raw = json.loads(self.term_inventory_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Term inventory config must be a JSON object: {self.term_inventory_path}")

        mode = str(raw.get("mode", "merge")).strip().lower()
        if mode not in TERM_INVENTORY_MODES:
            raise ValueError(f"Unsupported term inventory mode `{mode}`. Use one of: {sorted(TERM_INVENTORY_MODES)}")

        config_inventory = _term_inventory_from_config(raw, source=str(self.term_inventory_path))
        if mode == "replace":
            return config_inventory
        return _merge_term_inventory(generated, config_inventory, source=f"generated+{self.term_inventory_path}")

    def find_terms(self, query: str) -> list[str]:
        normalized = clean_text(query).lower()
        found: list[str] = []
        for term in self.terms:
            if term.lower() in normalized and term not in found:
                found.append(term)
        return found

    def search(self, query: str, top_k: int = 6, prefer_terms: Optional[list[str]] = None) -> list[SourceChunk]:
        query_clean = clean_text(query)
        query_tokens = tokenize(query_clean)
        prefer_terms = prefer_terms or self.find_terms(query_clean)
        scored: list[SourceChunk] = []

        for chunk in self.chunks:
            score = 0.0
            title_lower = chunk.title.lower()
            text_lower = chunk.text.lower()

            for term in prefer_terms:
                term_lower = term.lower()
                if term_lower and term_lower in title_lower:
                    score += 12.0
                elif term_lower and term_lower in text_lower:
                    score += 5.0

            title_tokens = tokenize(chunk.title)
            text_tokens = tokenize(chunk.text[:2500])
            score += len(query_tokens & title_tokens) * 2.0
            score += len(query_tokens & text_tokens) * 0.25

            if chunk.source.endswith("dic.md"):
                score -= 0.4
            if chunk.layer == "interaction_mechanism":
                score += 0.3

            if score > 0:
                scored.append(
                    SourceChunk(
                        id=chunk.id,
                        title=chunk.title,
                        source=chunk.source,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        text=chunk.text,
                        layer=chunk.layer,
                        terms=chunk.terms,
                        score=round(score, 3),
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe(scored)[:top_k]

    def _dedupe(self, chunks: list[SourceChunk]) -> list[SourceChunk]:
        deduped: list[SourceChunk] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            key = (chunk.layer, strip_heading_prefix(chunk.title))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped


def _canonical_output_term(term: str) -> str:
    canonical = compact_whitespace(term).strip("* ")
    if not canonical or len(canonical) > 32:
        return ""
    if re.match(r"^\d", canonical):
        return ""
    if TERM_CODE_RE.fullmatch(canonical):
        return ""
    if re.match(r"^[a-z]\s+", canonical, flags=re.IGNORECASE):
        return ""
    if "vs" in canonical.lower():
        return ""
    return canonical


def _term_inventory_from_config(raw: dict[str, Any], *, source: str) -> TermInventory:
    structural_terms = _normalize_terms(raw.get("structural_terms", STRUCTURAL_TERMS))
    by_layer_raw = raw.get("by_layer", {})
    if not isinstance(by_layer_raw, dict):
        raise ValueError("Term inventory `by_layer` must be an object.")
    by_type_raw = raw.get("by_type", raw.get("term_types", {}))
    if not isinstance(by_type_raw, dict):
        raise ValueError("Term inventory `by_type` must be an object.")

    by_layer: dict[Layer, list[str]] = {}
    valid_layers = set(Layer.__args__)  # type: ignore[attr-defined]
    for layer, terms in by_layer_raw.items():
        if layer not in valid_layers:
            raise ValueError(f"Unknown term inventory layer `{layer}`.")
        by_layer[layer] = _normalize_terms(terms)

    by_type = {str(term_type).strip(): _normalize_terms(terms) for term_type, terms in by_type_raw.items() if str(term_type).strip()}
    return TermInventory(by_layer=by_layer, by_type=by_type, structural_terms=structural_terms, source=source)


def _merge_term_inventory(generated: TermInventory, custom: TermInventory, *, source: str) -> TermInventory:
    by_layer: dict[Layer, list[str]] = {}
    for layer in set(generated.by_layer) | set(custom.by_layer):
        by_layer[layer] = _dedupe_terms(custom.by_layer.get(layer, []) + generated.by_layer.get(layer, []))
    by_type: dict[str, list[str]] = {}
    for term_type in set(generated.by_type) | set(custom.by_type):
        by_type[term_type] = _dedupe_terms(custom.by_type.get(term_type, []) + generated.by_type.get(term_type, []))
    return TermInventory(
        by_layer=by_layer,
        by_type=by_type,
        structural_terms=_dedupe_terms(custom.structural_terms + generated.structural_terms),
        source=source,
    )


def _normalize_terms(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Term inventory term lists must be arrays.")
    return _dedupe_terms(str(item).strip() for item in value if str(item).strip())


def _dedupe_terms(items) -> list[str]:
    terms: list[str] = []
    for item in items:
        if item and item not in terms:
            terms.append(item)
    return terms
