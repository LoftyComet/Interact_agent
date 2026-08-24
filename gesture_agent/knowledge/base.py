from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

from gesture_agent.core.models import Layer, SourceChunk, StructuredKnowledgeItem, TermInventory
from gesture_agent.core.text_utils import clean_text, compact_whitespace, strip_heading_prefix, tokenize
from gesture_agent.retrieval import HybridRetriever
from gesture_agent.knowledge.mechanism_registry import MechanismRegistry


MECHANISM_HEADING_RE = re.compile(r"^\d+-[a-z]\s+[^、，,]{1,80}$", re.IGNORECASE)
PROPERTY_HEADING_RE = re.compile(r"^[（(]\d+[）)][^,，。；;]{0,30}(?:属性|信号|阶次控制)[^,，。；;]{0,40}$")
FORM_HEADING_RE = re.compile(r"^[（(]\d+[）)][^：:]{1,24}$")
TOP_HEADING_RE = re.compile(r"^\d+\.\s+[^：:]{2,40}$")
COMPARISON_FILENAME = "交互机制对比.md"
COMPARISON_HEADING_RE = re.compile(r"^\d+、.{2,80}$")
COMPARISON_SPLIT_RE = re.compile(r"\s*(?:vs|VS|Vs|和|与|、|/)\s*")
TERM_CODE_RE = re.compile(r"^(?:\d+-[a-z]|[0-9A-Za-z /×÷+\-]+)$", re.IGNORECASE)

# Retrieval scoring weights
SCORE_TERM_IN_TITLE = 12.0
SCORE_TERM_IN_TEXT = 5.0
SCORE_QUERY_TOKEN_IN_TITLE = 2.0
SCORE_QUERY_TOKEN_IN_BODY = 0.25
SCORE_JACCARD_MULTIPLIER = 8.0
SCORE_LAYER_MECHANISM_BONUS = 0.3
SCORE_STRUCTURED_EXACT_MATCH = 8.0
SCORE_STRUCTURED_RELATED_MATCH = 3.0
SCORE_STRUCTURED_TYPE_MATCH = 1.5
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
DEFAULT_STRUCTURED_KNOWLEDGE_FILENAME = "structured_knowledge.json"
TERM_INVENTORY_MODES = {"merge", "replace"}
DEFAULT_ALIASES = {
    "点一下": "单击",
    "点击一下": "单击",
    "按一下": "按下",
    "按住": "长按",
    "长摁": "长按",
    "拖动": "拖拽",
    "滑一下": "滑动",
    "双点": "双击",
    "连点两下": "双击",
    "knob": "旋钮",
    "slider": "滑块",
}


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        term_inventory_path: Optional[Union[str, Path]] = None,
        structured_knowledge_path: Optional[Union[str, Path]] = None,
        index_dir: Optional[Union[str, Path]] = None,
        embedder: Any = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.term_inventory_path = Path(term_inventory_path) if term_inventory_path else self.data_dir / DEFAULT_TERM_INVENTORY_FILENAME
        self.structured_knowledge_path = (
            Path(structured_knowledge_path)
            if structured_knowledge_path
            else self.data_dir / DEFAULT_STRUCTURED_KNOWLEDGE_FILENAME
        )
        self.chunks: list[SourceChunk] = []
        self.terms: list[str] = []
        self.structured_items: list[StructuredKnowledgeItem] = []
        self.term_inventory = TermInventory()
        self.mechanism_registry = MechanismRegistry(())
        self.retriever: Optional[HybridRetriever] = None
        self.index_chunk_count = 0
        self._index_dir = Path(index_dir) if index_dir else None
        self._embedder = embedder

    @classmethod
    def load(
        cls,
        data_dir: Union[str, Path] = "data",
        term_inventory_path: Optional[Union[str, Path]] = None,
        structured_knowledge_path: Optional[Union[str, Path]] = None,
        index_dir: Optional[Union[str, Path]] = None,
        embedder: Any = None,
    ) -> "KnowledgeBase":
        kb = cls(
            data_dir,
            term_inventory_path=term_inventory_path,
            structured_knowledge_path=structured_knowledge_path,
            index_dir=index_dir,
            embedder=embedder,
        )
        kb._load()
        if kb._index_dir is not None:
            kb.attach_index(kb._index_dir, embedder=kb._embedder)
        return kb

    def attach_index(self, index_dir: Union[str, Path], *, embedder: Any = None) -> None:
        """Switch search to the deployable index while preserving legacy metadata."""

        self.retriever = HybridRetriever.from_directory(index_dir, embedder=embedder)
        manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
        self.index_chunk_count = int(manifest.get("chunk_count", 0))

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
        self.mechanism_registry = MechanismRegistry.load(self.term_inventory_path)
        term_set.update(self.term_inventory.all_terms())
        self.terms = sorted(term_set, key=lambda item: (-len(item), item))
        generated_structured = self._build_structured_items(chunks)
        self.structured_items = self._load_structured_knowledge_config(generated_structured)
        self._apply_structured_items()

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
            if filename == "序篇_基础属性.md" and PROPERTY_HEADING_RE.match(stripped):
                indexes.append(idx)
                continue
            if filename == "控件形态与含义.md" and FORM_HEADING_RE.match(stripped):
                indexes.append(idx)
                continue
            if TOP_HEADING_RE.match(stripped) and filename in {"序篇_基础属性.md", "交互机制.md", "控件形态与含义.md"}:
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
        if filename == "控件形态与含义.md" and FORM_HEADING_RE.match(title):
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

    def _build_structured_items(self, chunks: list[SourceChunk]) -> list[StructuredKnowledgeItem]:
        items: list[StructuredKnowledgeItem] = []
        for chunk in chunks:
            term = _canonical_output_term(strip_heading_prefix(chunk.title))
            if not term and chunk.terms:
                term = _canonical_output_term(chunk.terms[0])
            if not term:
                continue
            items.append(
                StructuredKnowledgeItem(
                    id=chunk.id,
                    term=term,
                    term_type=_term_type_for_layer(chunk.layer, chunk.title),
                    layer=chunk.layer,
                    definition=_first_meaningful_line(chunk.text),
                    aliases=[alias for alias, canonical in DEFAULT_ALIASES.items() if canonical == term],
                    properties=_terms_for_layer(self.term_inventory, "basic_property", chunk.text),
                    mechanisms=_terms_for_layer(self.term_inventory, "interaction_mechanism", chunk.text),
                    control_forms=_terms_for_layer(self.term_inventory, "control_form", chunk.text),
                    response_logic=_extract_response_logic(chunk.text),
                    related_terms=[item for item in chunk.terms if item != term],
                    source=chunk.citation(),
                    evidence=chunk.text[:800],
                )
            )
        return items

    def _load_structured_knowledge_config(
        self,
        generated: list[StructuredKnowledgeItem],
    ) -> list[StructuredKnowledgeItem]:
        if not self.structured_knowledge_path.exists():
            return generated

        raw = json.loads(self.structured_knowledge_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw_items = raw.get("items", [])
        else:
            raw_items = raw
        if not isinstance(raw_items, list):
            raise ValueError(f"Structured knowledge config must be an array or an object with `items`: {self.structured_knowledge_path}")

        custom_items = [_structured_item_from_config(item, source=str(self.structured_knowledge_path)) for item in raw_items]
        generated_by_term = {item.term: item for item in generated}
        merged: list[StructuredKnowledgeItem] = []
        seen: set[str] = set()
        for item in custom_items:
            base = generated_by_term.get(item.term)
            merged_item = _merge_structured_item(base, item) if base else item
            merged.append(merged_item)
            seen.add(item.term)
        for item in generated:
            if item.term not in seen:
                merged.append(item)
        return merged

    def _apply_structured_items(self) -> None:
        term_set = set(self.terms)
        aliases = {**DEFAULT_ALIASES, **self.term_inventory.aliases}
        for item in self.structured_items:
            values = [
                item.term,
                *item.aliases,
                *item.properties,
                *item.mechanisms,
                *item.control_forms,
                *item.related_terms,
            ]
            for value in values:
                if value:
                    term_set.add(value)
            for alias in item.aliases:
                aliases[alias] = item.term
        self.term_inventory.aliases = {alias: canonical for alias, canonical in aliases.items() if alias and canonical}
        term_set.update(self.term_inventory.all_terms())
        self.terms = sorted(term_set, key=lambda item: (-len(item), item))

    def export_structured_knowledge(self, output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        payload = {
            "items": [asdict(item) for item in self.structured_items],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def normalize_query(self, query: str) -> str:
        rewritten = clean_text(query)
        for alias, canonical in sorted(self.term_inventory.aliases.items(), key=lambda item: (-len(item[0]), item[0])):
            if alias and canonical and alias.lower() in rewritten.lower():
                rewritten = _replace_case_insensitive(rewritten, alias, canonical)
        return compact_whitespace(rewritten)

    def rewrite_query(self, query: str, prefer_terms: Optional[list[str]] = None) -> str:
        normalized = self.normalize_query(query)
        terms = _dedupe_terms((prefer_terms or []) + self.find_terms(normalized))
        expansions: list[str] = []
        related_to_expand: list[str] = []
        for item in self.structured_items:
            if item.term not in terms:
                continue
            expansions.extend([item.term, item.term_type, *item.aliases, *item.properties, *item.mechanisms, *item.control_forms, *item.related_terms])
            if item.response_logic:
                expansions.append(item.response_logic)
            related_to_expand.extend(item.related_terms)
        # Expand one level of related_terms
        for item in self.structured_items:
            if item.term in related_to_expand and item.term not in terms:
                expansions.extend([item.term, *item.aliases, *item.properties, *item.mechanisms])
        if not expansions:
            return normalized
        return compact_whitespace(normalized + " " + " ".join(_dedupe_terms(expansions)))

    def find_terms(self, query: str) -> list[str]:
        normalized = self.normalize_query(query).lower()
        found: list[str] = []
        for alias, canonical in sorted(self.term_inventory.aliases.items(), key=lambda item: (-len(item[0]), item[0])):
            if alias.lower() in normalized and canonical not in found:
                found.append(canonical)
        for term in self.terms:
            canonical = self.term_inventory.aliases.get(term, term)
            if term.lower() in normalized and canonical not in found:
                found.append(canonical)
        return found

    def search(self, query: str, top_k: int = 6, prefer_terms: Optional[list[str]] = None) -> list[SourceChunk]:
        if self.retriever is not None:
            results = self.retriever.retrieve(query, top_k=top_k, preferred_terms=prefer_terms)
            return [self._retrieval_result_to_source_chunk(result) for result in results]
        normalized_query = self.normalize_query(query)
        prefer_terms = _dedupe_terms((prefer_terms or []) + self.find_terms(normalized_query))
        query_clean = self.rewrite_query(normalized_query, prefer_terms=prefer_terms)
        query_tokens = tokenize(query_clean)
        scored: list[SourceChunk] = []

        for chunk in self.chunks:
            keyword_score = 0.0
            title_lower = chunk.title.lower()
            text_lower = chunk.text.lower()
            chunk_tokens = self._chunk_tokens(chunk)

            for term in prefer_terms:
                term_lower = term.lower()
                if term_lower and term_lower in title_lower:
                    keyword_score += SCORE_TERM_IN_TITLE
                elif term_lower and term_lower in text_lower:
                    keyword_score += SCORE_TERM_IN_TEXT

            title_tokens = tokenize(chunk.title)
            keyword_score += len(query_tokens & title_tokens) * SCORE_QUERY_TOKEN_IN_TITLE
            keyword_score += len(query_tokens & chunk_tokens) * SCORE_QUERY_TOKEN_IN_BODY

            if chunk.layer == "interaction_mechanism":
                keyword_score += SCORE_LAYER_MECHANISM_BONUS

            vector_score = _jaccard_similarity(query_tokens, chunk_tokens)
            structured_score = self._structured_score(chunk, prefer_terms, query_tokens)
            score = keyword_score + vector_score * SCORE_JACCARD_MULTIPLIER + structured_score

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

    def _retrieval_result_to_source_chunk(self, result: Any) -> SourceChunk:
        positions: list[int] = []
        for locator in result.source_locators:
            for key in ("line_start", "page", "paragraph", "table"):
                value = locator.get(key)
                if isinstance(value, int):
                    positions.append(value + (1 if key in {"paragraph", "table"} else 0))
                    break
        start = min(positions) if positions else 1
        end = max(positions) if positions else start
        layer: Layer = "unknown"
        result_terms = set(result.terms)
        heading_text = " ".join([result.title, *result.heading_path])
        best_layer_score = 0
        for candidate_layer, terms in self.term_inventory.by_layer.items():
            layer_score = len(result_terms.intersection(terms))
            layer_score += sum(3 for term in terms if term and term in heading_text)
            if layer_score > best_layer_score:
                layer = candidate_layer
                best_layer_score = layer_score
        return SourceChunk(
            id=result.chunk_id,
            title=result.title,
            source=result.source_path,
            start_line=start,
            end_line=end,
            text=result.context_text,
            layer=layer,
            terms=result.terms,
            score=result.score,
        )

    def _chunk_tokens(self, chunk: SourceChunk) -> set[str]:
        item = self._structured_item_for_chunk(chunk)
        structured_text = ""
        if item:
            structured_text = " ".join(
                [
                    item.term,
                    item.term_type,
                    *item.aliases,
                    *item.properties,
                    *item.mechanisms,
                    *item.control_forms,
                    *item.related_terms,
                    item.definition,
                    item.response_logic,
                ]
            )
        return tokenize(f"{chunk.title}\n{chunk.text[:2500]}\n{structured_text}")

    def _structured_score(self, chunk: SourceChunk, prefer_terms: list[str], query_tokens: set[str]) -> float:
        item = self._structured_item_for_chunk(chunk)
        if item is None:
            return 0.0
        score = 0.0
        structured_terms = _dedupe_terms(
            [
                item.term,
                *item.aliases,
                *item.properties,
                *item.mechanisms,
                *item.control_forms,
                *item.related_terms,
            ]
        )
        for term in prefer_terms:
            if term == item.term:
                score += SCORE_STRUCTURED_EXACT_MATCH
            elif term in structured_terms:
                score += SCORE_STRUCTURED_RELATED_MATCH
        if item.term_type and tokenize(item.term_type) & query_tokens:
            score += SCORE_STRUCTURED_TYPE_MATCH
        return score

    def _structured_item_for_chunk(self, chunk: SourceChunk) -> Optional[StructuredKnowledgeItem]:
        clean_title = strip_heading_prefix(chunk.title)
        for item in self.structured_items:
            if item.id == chunk.id or item.term == clean_title or item.term in chunk.terms:
                return item
        return None

    def _dedupe(self, chunks: list[SourceChunk]) -> list[SourceChunk]:
        # Input must be sorted by score descending so the first occurrence is the highest-scoring one.
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


def _term_type_for_layer(layer: Layer, title: str) -> str:
    if layer == "basic_property":
        return "基础属性"
    if layer == "control_form":
        return "控件形态"
    if layer == "interaction_mechanism":
        if re.match(r"^[34]-[a-z]", title, flags=re.IGNORECASE):
            return "交互逻辑的组合与扩展"
        return "基础交互逻辑"
    if layer == "multimodal_interaction":
        return "含义与多模态"
    if layer == "voice_interaction":
        return "语音交互"
    if layer == "background_knowledge":
        return "背景概念"
    return "待定"


def _first_meaningful_line(text: str) -> str:
    for line in text.splitlines()[1:]:
        line = compact_whitespace(line).strip("-* ")
        if line and not line.startswith("##") and not line.startswith("#"):
            return line[:300]
    return compact_whitespace(text)[:300]


def _terms_for_layer(inventory: TermInventory, layer: Layer, text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in inventory.by_layer.get(layer, []) if term.lower() in lowered][:12]


def _extract_response_logic(text: str) -> str:
    lines = [compact_whitespace(line).strip("-* ") for line in text.splitlines()]
    for idx, line in enumerate(lines):
        if any(key in line for key in ["响应逻辑", "反馈", "状态", "变化"]):
            window = " ".join(item for item in lines[idx : idx + 3] if item)
            return window[:300]
    return ""


def _structured_item_from_config(raw: Any, *, source: str) -> StructuredKnowledgeItem:
    if not isinstance(raw, dict):
        raise ValueError(f"Structured knowledge item must be an object: {source}")
    term = str(raw.get("term", "")).strip()
    if not term:
        raise ValueError(f"Structured knowledge item is missing `term`: {source}")
    layer = str(raw.get("layer", "unknown"))
    valid_layers = set(Layer.__args__)  # type: ignore[attr-defined]
    if layer not in valid_layers:
        raise ValueError(f"Unknown structured knowledge layer `{layer}`.")
    return StructuredKnowledgeItem(
        id=str(raw.get("id") or f"structured:{term}"),
        term=term,
        term_type=str(raw.get("term_type") or _term_type_for_layer(layer, "")),
        layer=layer,  # type: ignore[arg-type]
        definition=str(raw.get("definition") or ""),
        aliases=_normalize_terms(raw.get("aliases", [])),
        properties=_normalize_terms(raw.get("properties", [])),
        mechanisms=_normalize_terms(raw.get("mechanisms", [])),
        control_forms=_normalize_terms(raw.get("control_forms", [])),
        response_logic=str(raw.get("response_logic") or ""),
        related_terms=_normalize_terms(raw.get("related_terms", [])),
        source=str(raw.get("source") or source),
        evidence=str(raw.get("evidence") or ""),
    )


def _merge_structured_item(
    base: Optional[StructuredKnowledgeItem],
    custom: StructuredKnowledgeItem,
) -> StructuredKnowledgeItem:
    if base is None:
        return custom
    return StructuredKnowledgeItem(
        id=custom.id or base.id,
        term=custom.term or base.term,
        term_type=custom.term_type or base.term_type,
        layer=custom.layer or base.layer,
        definition=custom.definition or base.definition,
        aliases=_dedupe_terms(custom.aliases + base.aliases),
        properties=_dedupe_terms(custom.properties + base.properties),
        mechanisms=_dedupe_terms(custom.mechanisms + base.mechanisms),
        control_forms=_dedupe_terms(custom.control_forms + base.control_forms),
        response_logic=custom.response_logic or base.response_logic,
        related_terms=_dedupe_terms(custom.related_terms + base.related_terms),
        source=custom.source or base.source,
        evidence=custom.evidence or base.evidence,
    )


def _replace_case_insensitive(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def _term_inventory_from_tree(raw: dict[str, Any], *, source: str) -> TermInventory:
    """Flatten a nested outline tree into a TermInventory.

    Each node may carry `layer` and `type`, inherited by descendants. Leaf nodes
    (no `children`) contribute their `label` to `by_layer[layer]` and `by_type[type]`.
    Node/leaf `aliases` map onto the canonical `label`.
    """
    structural_terms = _normalize_terms(raw.get("structural_terms", STRUCTURAL_TERMS))
    valid_layers = set(Layer.__args__)  # type: ignore[attr-defined]
    by_layer: dict[Layer, list[str]] = {}
    by_type: dict[str, list[str]] = {}
    subgroup_labels: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}

    def _push(bucket: dict, key: Any, term: str) -> None:
        if not key:
            return
        terms = bucket.setdefault(key, [])
        if term not in terms:
            terms.append(term)

    def _walk(node: Any, layer: Optional[str], term_type: Optional[str]) -> None:
        if not isinstance(node, dict):
            return
        layer = node.get("layer", layer)
        if layer is not None and layer not in valid_layers:
            raise ValueError(f"Unknown term inventory layer `{layer}` in tree.")
        term_type = node.get("type", term_type)
        for alias in node.get("aliases", []):
            a = str(alias).strip()
            label = str(node.get("label", "")).strip()
            if a and label:
                aliases[a] = label
        children = node.get("children")
        if isinstance(children, list) and children:
            # 中间节点（有 children 且不是根）→ 收集为子分组标签
            label = str(node.get("label", "")).strip()
            if label and term_type and label != raw.get("root", {}).get("label"):
                _push(subgroup_labels, str(term_type).strip(), label)
            for child in children:
                _walk(child, layer, term_type)
            return
        label = str(node.get("label", "")).strip()
        if label:
            _push(by_layer, layer, label)
            _push(by_type, str(term_type).strip() if term_type else "", label)

    _walk(raw.get("root", {}), None, None)
    by_type.pop("", None)

    aliases_raw = raw.get("aliases", {})
    if isinstance(aliases_raw, dict):
        for key, value in aliases_raw.items():
            if isinstance(value, dict):
                for alias, canonical in value.items():
                    a, c = str(alias).strip(), str(canonical).strip()
                    if a and c:
                        aliases[a] = c
            else:
                a, c = str(key).strip(), str(value).strip()
                if a and c:
                    aliases[a] = c

    return TermInventory(by_layer=by_layer, by_type=by_type, aliases=aliases, structural_terms=structural_terms, subgroup_labels=subgroup_labels, source=source)


def _term_inventory_from_config(raw: dict[str, Any], *, source: str) -> TermInventory:
    if str(raw.get("schema", "")).strip().lower() == "tree" or isinstance(raw.get("root"), dict):
        return _term_inventory_from_tree(raw, source=source)
    structural_terms = _normalize_terms(raw.get("structural_terms", STRUCTURAL_TERMS))
    by_layer_raw = raw.get("by_layer", {})
    if not isinstance(by_layer_raw, dict):
        raise ValueError("Term inventory `by_layer` must be an object.")
    by_type_raw = raw.get("by_type", raw.get("term_types", {}))
    if not isinstance(by_type_raw, dict):
        raise ValueError("Term inventory `by_type` must be an object.")
    aliases_raw = raw.get("aliases", {})
    if not isinstance(aliases_raw, dict):
        raise ValueError("Term inventory `aliases` must be an object.")

    by_layer: dict[Layer, list[str]] = {}
    valid_layers = set(Layer.__args__)  # type: ignore[attr-defined]
    for layer, terms in by_layer_raw.items():
        if layer not in valid_layers:
            raise ValueError(f"Unknown term inventory layer `{layer}`.")
        by_layer[layer] = _normalize_terms(terms)

    by_type = {str(term_type).strip(): _normalize_terms(terms) for term_type, terms in by_type_raw.items() if str(term_type).strip()}
    aliases: dict[str, str] = {}
    for key, value in aliases_raw.items():
        if isinstance(value, dict):
            for alias, canonical in value.items():
                a, c = str(alias).strip(), str(canonical).strip()
                if a and c:
                    aliases[a] = c
        else:
            a, c = str(key).strip(), str(value).strip()
            if a and c:
                aliases[a] = c
    return TermInventory(by_layer=by_layer, by_type=by_type, aliases=aliases, structural_terms=structural_terms, source=source)


def _merge_term_inventory(generated: TermInventory, custom: TermInventory, *, source: str) -> TermInventory:
    by_layer: dict[Layer, list[str]] = {}
    for layer in set(generated.by_layer) | set(custom.by_layer):
        by_layer[layer] = _dedupe_terms(custom.by_layer.get(layer, []) + generated.by_layer.get(layer, []))
    by_type: dict[str, list[str]] = {}
    for term_type in set(generated.by_type) | set(custom.by_type):
        by_type[term_type] = _dedupe_terms(custom.by_type.get(term_type, []) + generated.by_type.get(term_type, []))
    subgroup_labels: dict[str, list[str]] = {}
    for term_type in set(generated.subgroup_labels) | set(custom.subgroup_labels):
        subgroup_labels[term_type] = _dedupe_terms(custom.subgroup_labels.get(term_type, []) + generated.subgroup_labels.get(term_type, []))
    return TermInventory(
        by_layer=by_layer,
        by_type=by_type,
        aliases={**generated.aliases, **custom.aliases},
        structural_terms=_dedupe_terms(custom.structural_terms + generated.structural_terms),
        subgroup_labels=subgroup_labels,
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
