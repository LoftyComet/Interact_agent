from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

from gesture_agent.core.models import Layer, SourceChunk
from gesture_agent.core.text_utils import clean_text, compact_whitespace, strip_heading_prefix, tokenize


MECHANISM_HEADING_RE = re.compile(r"^\d+-[a-z]\s+[^、，,]{1,80}$", re.IGNORECASE)
PROPERTY_HEADING_RE = re.compile(r"^[（(]\d+[）)][^,，。；;]{0,30}(?:属性|信号|阶次控制)[^,，。；;]{0,40}$")
FORM_HEADING_RE = re.compile(r"^[（(]\d+[）)][^：:]{1,24}$")
TOP_HEADING_RE = re.compile(r"^\d+\.\s+[^：:]{2,40}$")
COMPARISON_FILENAME = "交互机制对比.md"
COMPARISON_HEADING_RE = re.compile(r"^\d+、.{2,80}$")
COMPARISON_SPLIT_RE = re.compile(r"\s*(?:vs|VS|Vs|和|与|、|/)\s*")


class KnowledgeBase:
    def __init__(self, data_dir: Union[str, Path] = "data") -> None:
        self.data_dir = Path(data_dir)
        self.chunks: list[SourceChunk] = []
        self.terms: list[str] = []

    @classmethod
    def load(cls, data_dir: Union[str, Path] = "data") -> "KnowledgeBase":
        kb = cls(data_dir)
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
