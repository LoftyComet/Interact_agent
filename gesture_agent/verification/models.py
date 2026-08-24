from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ..core.models import QuestionStructure


@dataclass
class TermIssue:
    original: str
    canonical: str
    issue_type: Literal["wrong_term", "impossible_combination", "contradictory", "ambiguous"]
    explanation: str


@dataclass
class InputVerificationResult:
    status: Literal["pass", "corrected", "needs_clarification"]
    issues: list[TermIssue] = field(default_factory=list)
    corrected_structure: Optional[QuestionStructure] = None
    correction_summary: str = ""


@dataclass
class OutputIssue:
    issue_type: Literal[
        "missing_section",
        "duplicate_section",
        "empty_section",
        "missing_direct_answer",
        "invalid_citation",
        "invalid_term",
        "knowledge_conflict",
        "format_error",
        "missing_mechanism_code",
        "mechanism_code_mismatch",
        "mechanism_name_mismatch",
        "unknown_mechanism_code",
    ]
    location: str
    description: str
    severity: Literal["error", "warning"] = "warning"


@dataclass
class OutputVerificationResult:
    status: Literal["pass", "issues_found"]
    issues: list[OutputIssue] = field(default_factory=list)
    should_retry: bool = False
    correction_hints: str = ""
