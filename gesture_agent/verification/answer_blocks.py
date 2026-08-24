from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal


AnswerBlockType = Literal["corpus_evidence", "design_reasoning", "conversation_response"]

CORPUS_EVIDENCE = "corpus_evidence"
DESIGN_REASONING = "design_reasoning"
CONVERSATION_RESPONSE = "conversation_response"

BLOCK_LABELS: dict[AnswerBlockType, str] = {
    CORPUS_EVIDENCE: "语料依据",
    DESIGN_REASONING: "设计推导（仅供参考）",
    CONVERSATION_RESPONSE: "对话回应",
}

_MARKER_RE = re.compile(
    r"<!--\s*ixdl-answer-block:(corpus_evidence|design_reasoning|conversation_response)\s*-->",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"(?<!!)\[(\d+)\](?!\()")


@dataclass(frozen=True)
class AnswerBlock:
    """One provenance-aware portion of a generated answer."""

    type: AnswerBlockType
    markdown: str
    citation_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def title(self) -> str:
        return BLOCK_LABELS[self.type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "title": self.title,
            "markdown": self.markdown,
            "citations": list(self.citation_ids),
        }


@dataclass(frozen=True)
class AnswerBlockDocument:
    """Parsed answer blocks while retaining legacy Markdown compatibility."""

    blocks: tuple[AnswerBlock, ...]
    explicit_markers: bool = False

    def blocks_of_type(self, block_type: AnswerBlockType) -> tuple[AnswerBlock, ...]:
        return tuple(block for block in self.blocks if block.type == block_type)

    def markdown_for(self, block_type: AnswerBlockType) -> str:
        return "\n\n".join(
            block.markdown.strip()
            for block in self.blocks_of_type(block_type)
            if block.markdown.strip()
        )

    @property
    def corpus_markdown(self) -> str:
        return self.markdown_for(CORPUS_EVIDENCE)

    def replace_corpus_markdown(self, markdown: str) -> "AnswerBlockDocument":
        replacement = _make_block(CORPUS_EVIDENCE, markdown)
        updated: list[AnswerBlock] = []
        inserted = False
        for block in self.blocks:
            if block.type != CORPUS_EVIDENCE:
                updated.append(block)
                continue
            if not inserted and markdown.strip():
                updated.append(replacement)
                inserted = True
        if not inserted and markdown.strip():
            updated.insert(0, replacement)
        return replace(self, blocks=tuple(updated))

    def render_markdown(self) -> str:
        if not self.explicit_markers and len(self.blocks) == 1:
            return self.blocks[0].markdown.strip()
        parts: list[str] = []
        for block in self.blocks:
            if not block.markdown.strip():
                continue
            parts.append(f"<!-- ixdl-answer-block:{block.type} -->\n\n{block.markdown.strip()}")
        return "\n\n".join(parts).strip()

    def to_list(self) -> list[dict[str, Any]]:
        return [block.to_dict() for block in self.blocks if block.markdown.strip()]


def parse_answer_blocks(
    markdown: str,
    *,
    default_type: AnswerBlockType = CORPUS_EVIDENCE,
) -> AnswerBlockDocument:
    """Parse invisible provenance markers from Markdown.

    Unmarked legacy answers remain valid and are treated as corpus evidence. This
    lets the API evolve without breaking existing clients while new generations
    can label any non-source-backed design reasoning explicitly.
    """

    matches = list(_MARKER_RE.finditer(markdown))
    if not matches:
        block = _make_block(default_type, markdown)
        return AnswerBlockDocument(blocks=(block,) if block.markdown else ())

    blocks: list[AnswerBlock] = []
    prefix = markdown[: matches[0].start()].strip()
    if prefix:
        blocks.append(_make_block(default_type, prefix))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[match.end():end].strip()
        if not body:
            continue
        block_type = match.group(1).lower()
        blocks.append(_make_block(block_type, body))  # type: ignore[arg-type]

    return AnswerBlockDocument(blocks=tuple(_merge_adjacent(blocks)), explicit_markers=True)


def _make_block(block_type: AnswerBlockType, markdown: str) -> AnswerBlock:
    cleaned = markdown.strip()
    citations = tuple(dict.fromkeys(int(value) for value in _CITATION_RE.findall(cleaned)))
    return AnswerBlock(type=block_type, markdown=cleaned, citation_ids=citations)


def _merge_adjacent(blocks: list[AnswerBlock]) -> list[AnswerBlock]:
    merged: list[AnswerBlock] = []
    for block in blocks:
        if merged and merged[-1].type == block.type:
            merged[-1] = _make_block(
                block.type,
                f"{merged[-1].markdown}\n\n{block.markdown}",
            )
        else:
            merged.append(block)
    return merged
