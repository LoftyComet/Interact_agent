from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CITATION_RE = re.compile(r"(?<!!)\[(\d+)\](?!\()")
_BLOCK_MARKER_RE = re.compile(
    r"<!--\s*ixdl-answer-block:(?:corpus_evidence|design_reasoning|conversation_response)\s*-->",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerSection:
    """A named section in a user-facing answer."""

    title: str
    body: str


@dataclass(frozen=True)
class AnswerDocument:
    """Structured representation of the Markdown answer contract."""

    direct_answer: str
    sections: tuple[AnswerSection, ...]
    citation_ids: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnswerContractViolation:
    code: Literal[
        "missing_direct_answer",
        "missing_section",
        "duplicate_section",
        "empty_section",
        "format_error",
    ]
    location: str
    message: str


@dataclass(frozen=True)
class AnswerParseResult:
    document: AnswerDocument
    violations: tuple[AnswerContractViolation, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return not self.violations


def parse_answer_markdown(markdown: str, expected_sections: list[str]) -> AnswerParseResult:
    """Parse generated Markdown and validate its stable answer contract.

    Expected headings are matched exactly first. A decorated heading such as
    ``## 核心定义（概览）`` is also accepted when it contains one unambiguous
    expected title, then canonicalized by :func:`render_answer_markdown`.
    """

    stripped = markdown.strip()
    visible_markdown = _BLOCK_MARKER_RE.sub("", markdown)
    violations: list[AnswerContractViolation] = []
    if _is_whole_code_fence(stripped):
        violations.append(AnswerContractViolation(
            code="format_error",
            location="整体",
            message="输出被包裹在代码块中，应直接输出 Markdown",
        ))
    if stripped.startswith("{") and stripped.endswith("}"):
        violations.append(AnswerContractViolation(
            code="format_error",
            location="整体",
            message="输出是 JSON 对象，应输出面向阅读的 Markdown",
        ))

    matches = list(_HEADING_RE.finditer(visible_markdown))
    direct_answer = visible_markdown[: matches[0].start() if matches else len(visible_markdown)].strip()
    if not direct_answer or direct_answer.startswith("##"):
        violations.append(AnswerContractViolation(
            code="missing_direct_answer",
            location="开头",
            message="所有章节之前必须有一段直接回答用户问题的结论",
        ))

    parsed_sections: list[AnswerSection] = []
    canonical_counts = {title: 0 for title in expected_sections}
    for index, match in enumerate(matches):
        raw_title = match.group(1).strip()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(visible_markdown)
        body = visible_markdown[match.end():body_end].strip()
        canonical_title = _canonical_heading(raw_title, expected_sections)
        parsed_sections.append(AnswerSection(title=canonical_title or raw_title, body=body))
        if canonical_title is None:
            continue
        canonical_counts[canonical_title] += 1
        if canonical_counts[canonical_title] > 1:
            violations.append(AnswerContractViolation(
                code="duplicate_section",
                location=f"## {canonical_title}",
                message=f"章节「{canonical_title}」重复出现",
            ))
        if not body or _is_placeholder_body(body):
            violations.append(AnswerContractViolation(
                code="empty_section",
                location=f"## {canonical_title}",
                message=f"章节「{canonical_title}」没有正文内容",
            ))

    for title, count in canonical_counts.items():
        if count == 0:
            violations.append(AnswerContractViolation(
                code="missing_section",
                location=f"## {title}",
                message=f"缺少章节「{title}」",
            ))

    citations = tuple(dict.fromkeys(int(value) for value in _CITATION_RE.findall(markdown)))
    return AnswerParseResult(
        document=AnswerDocument(
            direct_answer=direct_answer,
            sections=tuple(parsed_sections),
            citation_ids=citations,
        ),
        violations=tuple(violations),
    )


def render_answer_markdown(document: AnswerDocument, expected_sections: list[str]) -> str:
    """Render an answer with deterministic heading names and section order."""

    by_title: dict[str, AnswerSection] = {}
    extras: list[AnswerSection] = []
    expected = set(expected_sections)
    for section in document.sections:
        if section.title in expected and section.title not in by_title:
            by_title[section.title] = section
        elif section.title not in expected:
            extras.append(section)

    ordered = [by_title[title] for title in expected_sections if title in by_title]
    ordered.extend(extras)
    parts = [document.direct_answer.strip()]
    for section in ordered:
        parts.append(f"## {section.title}\n\n{section.body.strip()}")
    return "\n\n".join(part for part in parts if part).strip()


def _canonical_heading(raw_title: str, expected_sections: list[str]) -> str | None:
    if raw_title in expected_sections:
        return raw_title
    matches = [title for title in expected_sections if title in raw_title]
    return matches[0] if len(matches) == 1 else None


def _is_whole_code_fence(markdown: str) -> bool:
    if not markdown.startswith("```"):
        return False
    lines = markdown.splitlines()
    return len(lines) >= 2 and lines[-1].strip() == "```"


def _is_placeholder_body(body: str) -> bool:
    visible = body.strip()
    return visible.endswith(("：", ":"))
