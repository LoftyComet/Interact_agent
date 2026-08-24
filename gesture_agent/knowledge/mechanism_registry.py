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

    def validate_answer(self, text: str) -> tuple[MechanismNamingIssue, ...]:
        """Require every mechanism mention to share a clause with its code.

        Longest-name matching avoids reporting ``开关`` again inside
        ``多点开关``. Codes found in the same clause are also checked against
        the mentioned Chinese alias or English name.
        """

        mentions = self._find_mentions(text)
        issues: list[MechanismNamingIssue] = []
        for start, end, mention, entry in mentions:
            clause = _surrounding_clause(text, start, end)
            found_codes = tuple(dict.fromkeys(
                match.group(1).lower() for match in _CODE_IN_TEXT_RE.finditer(clause)
            ))
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
            clause = _surrounding_clause(text, match.start(), match.end())
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


def _contains_name(text: str, name: str) -> bool:
    if not name:
        return False
    return name.lower() in text.lower()


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
