from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_MECHANISM_CODE_RE = re.compile(r"^[1-4]-[a-i]$", re.IGNORECASE)
_CODE_IN_TEXT_RE = re.compile(r"(?<![0-9A-Za-z-])(\d+-[a-z])(?![0-9A-Za-z-])", re.IGNORECASE)
_CLAUSE_BOUNDARY_RE = re.compile(r"[。！？；;\n]")


@dataclass(frozen=True)
class MechanismEntry:
    code: str
    label: str
    label_en: str
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(value for value in (self.label, *self.aliases, self.label_en) if value)


@dataclass(frozen=True)
class MechanismNamingIssue:
    issue_type: str
    mention: str
    expected: str
    found_codes: tuple[str, ...] = field(default_factory=tuple)


class MechanismRegistry:
    """Canonical IxDL mechanism code/name mapping and answer validator."""

    def __init__(self, entries: Iterable[MechanismEntry]) -> None:
        normalized = sorted(entries, key=lambda item: item.code)
        self.entries = tuple(normalized)
        self.by_code = {entry.code.lower(): entry for entry in normalized}

    @classmethod
    def load(cls, term_inventory_path: str | Path) -> "MechanismRegistry":
        path = Path(term_inventory_path)
        if not path.exists():
            return cls(())
        raw = json.loads(path.read_text(encoding="utf-8"))
        root = raw.get("root", raw) if isinstance(raw, dict) else {}
        entries: list[MechanismEntry] = []

        def visit(node: Any, inherited_layer: str = "") -> None:
            if not isinstance(node, dict):
                return
            layer = str(node.get("layer") or inherited_layer)
            code = str(node.get("code") or "").lower()
            label = str(node.get("label") or "").strip()
            if layer == "interaction_mechanism" and _MECHANISM_CODE_RE.fullmatch(code) and label:
                entries.append(MechanismEntry(
                    code=code,
                    label=label,
                    label_en=str(node.get("label_en") or "").strip(),
                    aliases=tuple(
                        str(value).strip()
                        for value in node.get("aliases", [])
                        if str(value).strip()
                    ),
                ))
            for child in node.get("children", []):
                visit(child, layer)

        visit(root)
        return cls(entries)

    def format_for_prompt(self) -> str:
        if not self.entries:
            return "未加载机制编号注册表；不要猜测编号或英文名。"
        return "；".join(
            f"{entry.code} {entry.label}（{entry.label_en}）"
            for entry in self.entries
        )

    def validate_answer(
        self,
        text: str,
        *,
        required_labels: Iterable[str] = (),
    ) -> tuple[MechanismNamingIssue, ...]:
        """Require the first formal mechanism mention to carry its code.

        Longest-name matching avoids reporting ``开关`` again inside
        ``多点开关``. Ordinary verb usage such as ``按下按钮`` is not treated
        as a named mechanism unless the question itself targets that mechanism.
        Codes found anywhere are still checked against names in their clause.
        """

        mentions = self._find_mentions(text)
        required = {str(value) for value in required_labels}
        issues: list[MechanismNamingIssue] = []
        checked_entries: set[str] = set()
        for start, end, mention, entry in mentions:
            if entry.code in checked_entries:
                continue
            is_required = bool(required.intersection({entry.label, *entry.aliases}))
            if not is_required and not _is_formal_mention(text, start, end, entry):
                continue
            checked_entries.add(entry.code)
            found_codes = _associated_codes(text, start, end)
            if entry.code in found_codes:
                continue
            issue_type = "mechanism_code_mismatch" if found_codes else "missing_mechanism_code"
            issues.append(MechanismNamingIssue(
                issue_type=issue_type,
                mention=mention,
                expected=f"{entry.code} {entry.label}（{entry.label_en}）",
                found_codes=found_codes,
            ))

        for match in _CODE_IN_TEXT_RE.finditer(text):
            code = match.group(1).lower()
            entry = self.by_code.get(code)
            if entry is None:
                issues.append(MechanismNamingIssue(
                    issue_type="unknown_mechanism_code",
                    mention=match.group(1),
                    expected="注册表中的 1-a 至 4-i 机制编号",
                    found_codes=(code,),
                ))
                continue
            clause = _nearby_text(text, match.start(), match.end())
            if not any(_contains_name(clause, name) for name in entry.names):
                other_names = [
                    other.label
                    for other in self.entries
                    if other.code != code and any(_contains_name(clause, name) for name in other.names)
                ]
                if other_names:
                    issues.append(MechanismNamingIssue(
                        issue_type="mechanism_name_mismatch",
                        mention=f"{code}+{other_names[0]}",
                        expected=f"{entry.code} {entry.label}（{entry.label_en}）",
                        found_codes=(code,),
                    ))
        return tuple(_dedupe_issues(issues))

    def normalize_answer(
        self,
        text: str,
        *,
        required_labels: Iterable[str] = (),
    ) -> str:
        """Canonicalize unambiguous codes and insert missing first-use codes."""

        required = {str(value) for value in required_labels}
        normalized = self._normalize_existing_codes(text)
        edits: list[tuple[int, str]] = []
        handled: set[str] = set()
        for start, end, _mention, entry in self._find_mentions(normalized):
            if entry.code in handled:
                continue
            is_required = bool(required.intersection({entry.label, *entry.aliases}))
            if not is_required and not _is_formal_mention(normalized, start, end, entry):
                continue
            handled.add(entry.code)
            found_codes = set(_associated_codes(normalized, start, end))
            if entry.code in found_codes or found_codes:
                continue
            edits.append((start, f"{entry.code} "))
        for start, insertion in sorted(edits, reverse=True):
            normalized = normalized[:start] + insertion + normalized[start:]
        return normalized

    def _normalize_existing_codes(self, text: str) -> str:
        """Repair a code when its nearest mechanism name makes the intent clear."""

        mentions = self._find_mentions(text)
        edits: list[tuple[int, int, str]] = []
        for match in _CODE_IN_TEXT_RE.finditer(text):
            nearby: list[tuple[int, MechanismEntry]] = []
            for start, end, _mention, entry in mentions:
                distance = max(start - match.end(), match.start() - end, 0)
                if distance <= 8:
                    nearby.append((distance, entry))
            if not nearby:
                continue
            minimum = min(distance for distance, _entry in nearby)
            closest_codes = {
                entry.code
                for distance, entry in nearby
                if distance == minimum
            }
            if len(closest_codes) != 1:
                continue
            expected = next(iter(closest_codes))
            if match.group(1).lower() != expected:
                edits.append((match.start(1), match.end(1), expected))

        normalized = text
        for start, end, replacement in sorted(edits, reverse=True):
            normalized = normalized[:start] + replacement + normalized[end:]
        return re.sub(
            r"(?<![0-9A-Za-z-])(\d+-[a-z])\s*[，,、/]\s*\1(?![0-9A-Za-z-])",
            r"\1",
            normalized,
            flags=re.IGNORECASE,
        )

    def _find_mentions(self, text: str) -> list[tuple[int, int, str, MechanismEntry]]:
        candidates: list[tuple[int, int, str, MechanismEntry]] = []
        for entry in self.entries:
            for name in entry.names:
                if not name:
                    continue
                if name == entry.label_en:
                    pattern = re.compile(
                        rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])",
                        re.IGNORECASE,
                    )
                else:
                    pattern = re.compile(re.escape(name))
                for match in pattern.finditer(text):
                    candidates.append((match.start(), match.end(), match.group(), entry))
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: list[tuple[int, int, str, MechanismEntry]] = []
        occupied: list[tuple[int, int]] = []
        for item in candidates:
            start, end = item[0], item[1]
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
            accepted.append(item)
        return accepted


def _surrounding_clause(text: str, start: int, end: int) -> str:
    left_matches = list(_CLAUSE_BOUNDARY_RE.finditer(text, 0, start))
    left = left_matches[-1].end() if left_matches else 0
    right_match = _CLAUSE_BOUNDARY_RE.search(text, end)
    right = right_match.start() if right_match else len(text)
    return text[left:right]


def _nearby_text(text: str, start: int, end: int, radius: int = 8) -> str:
    return text[max(0, start - radius):min(len(text), end + radius)]


def _associated_codes(text: str, start: int, end: int) -> tuple[str, ...]:
    before = text[max(0, start - 16):start]
    after = text[end:min(len(text), end + 16)]
    values: list[str] = []
    before_match = re.search(
        r"(?<![0-9A-Za-z-])(\d+-[a-z])\s*(?:\*\*)?\s*$",
        before,
        flags=re.IGNORECASE,
    )
    if before_match:
        values.append(before_match.group(1).lower())
    after_match = re.match(
        r"\s*(?:[（(]\s*)?(\d+-[a-z])(?![0-9A-Za-z-])",
        after,
        flags=re.IGNORECASE,
    )
    if after_match and after_match.group(1).lower() not in values:
        values.append(after_match.group(1).lower())
    return tuple(values)


def _contains_name(text: str, name: str) -> bool:
    if not name:
        return False
    return name.lower() in text.lower()


def _is_formal_mention(text: str, start: int, end: int, entry: MechanismEntry) -> bool:
    clause = _nearby_text(text, start, end)
    if _associated_codes(text, start, end):
        return True
    if entry.label_en and _contains_name(clause, entry.label_en):
        return True
    before = text[max(0, start - 32):start]
    after = text[end:min(len(text), end + 32)]
    context = before + text[start:end] + after
    return any(marker in context for marker in (
        "交互机制", "机制", "交互逻辑", "采用", "对应", "属于", "组合", "冲突调和",
    ))


def _dedupe_issues(issues: list[MechanismNamingIssue]) -> list[MechanismNamingIssue]:
    result: list[MechanismNamingIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        key = (issue.issue_type, issue.mention.lower(), issue.expected)
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result
