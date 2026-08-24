from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gesture_agent.knowledge import MechanismRegistry


_CITATION_RE = re.compile(r"(?<!!)\[(\d+)\](?!\()")


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


@dataclass(frozen=True)
class AnswerQualityScore:
    passed: bool
    score: float
    checks: tuple[QualityCheck, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                    "required": check.required,
                }
                for check in self.checks
            ],
        }


def score_api_response(
    case: dict[str, Any],
    response: dict[str, Any],
    mechanism_registry: MechanismRegistry,
) -> AnswerQualityScore:
    """Score deterministic parts of the IxDL answer contract.

    Semantic content coverage remains pending corpus audit. This scorer focuses
    on hard failures that should never require subjective judgement.
    """

    checks: list[QualityCheck] = []
    status = str(response.get("status") or "")
    checks.append(QualityCheck("answer_ready", status == "ready", f"status={status or 'missing'}"))
    error = response.get("error")
    checks.append(QualityCheck("provider_success", not error, str(error or "")))

    expected_intent = case.get("intent", {}).get("expected_runtime_label")
    expected_subtype = case.get("intent", {}).get("expected_subtype")
    structure = response.get("structure") or {}
    if expected_intent:
        actual_intent = structure.get("intent")
        checks.append(QualityCheck(
            "intent",
            actual_intent == expected_intent,
            f"expected={expected_intent}; actual={actual_intent}",
        ))
    if expected_subtype:
        actual_subtype = structure.get("subtype")
        checks.append(QualityCheck(
            "subtype",
            actual_subtype == expected_subtype,
            f"expected={expected_subtype}; actual={actual_subtype}",
        ))

    contract = case.get("response_contract", {})
    blocks = response.get("answer_blocks") or []
    block_types = [block.get("type") for block in blocks]
    checks.append(QualityCheck(
        "answer_blocks_present",
        bool(blocks),
        f"types={block_types}",
    ))
    for block_type in contract.get("required_ui_blocks", []):
        checks.append(QualityCheck(
            f"required_block:{block_type}",
            block_type in block_types,
            f"types={block_types}",
        ))
    allowed = set(contract.get("allowed_ui_blocks", []))
    unexpected = [value for value in block_types if allowed and value not in allowed]
    checks.append(QualityCheck(
        "allowed_block_types",
        not unexpected,
        f"unexpected={unexpected}",
    ))
    reasoning_allowed = bool(contract.get("reasoning_allowed"))
    checks.append(QualityCheck(
        "reasoning_policy",
        reasoning_allowed or "design_reasoning" not in block_types,
        "design_reasoning is forbidden for this case" if not reasoning_allowed else "",
    ))

    chunks = response.get("chunks") or []
    citation_errors: list[str] = []
    corpus_citations: list[int] = []
    for block in blocks:
        markdown = str(block.get("markdown") or "")
        parsed = list(dict.fromkeys(int(value) for value in _CITATION_RE.findall(markdown)))
        declared = [int(value) for value in block.get("citations", [])]
        if parsed != list(dict.fromkeys(declared)):
            citation_errors.append(f"{block.get('type')}: declared={declared}, parsed={parsed}")
        invalid = [value for value in parsed if value < 1 or value > len(chunks)]
        if invalid:
            citation_errors.append(f"{block.get('type')}: invalid={invalid}, chunks={len(chunks)}")
        if block.get("type") == "corpus_evidence":
            corpus_citations.extend(parsed)
    checks.append(QualityCheck(
        "citation_integrity",
        not citation_errors,
        "; ".join(citation_errors),
    ))
    if chunks and "corpus_evidence" in block_types:
        checks.append(QualityCheck(
            "corpus_citation_present",
            bool(corpus_citations),
            f"citations={corpus_citations}",
        ))

    answer = str(response.get("answer") or "")
    naming_issues = mechanism_registry.validate_answer(
        answer,
        required_labels=structure.get("terms") or [],
    )
    checks.append(QualityCheck(
        "mechanism_registry",
        not naming_issues,
        "; ".join(f"{issue.issue_type}:{issue.mention}" for issue in naming_issues),
    ))

    output_issues = response.get("output_issues") or []
    checks.append(QualityCheck(
        "output_verifier",
        not output_issues,
        "; ".join(str(value) for value in output_issues),
    ))
    fallback_applied = bool(response.get("safety_fallback_applied"))
    checks.append(QualityCheck(
        "no_safety_deletion",
        not fallback_applied,
        "unsupported claims were deleted after retries" if fallback_applied else "",
        required=False,
    ))
    grounding = response.get("grounding")
    if grounding is not None:
        grounding_status = grounding.get("status")
        checks.append(QualityCheck(
            "corpus_grounding",
            grounding_status == "pass",
            f"status={grounding_status}; score={grounding.get('score')}",
        ))
    reasoning_audit = response.get("reasoning_audit")
    if "design_reasoning" in block_types:
        reasoning_status = (reasoning_audit or {}).get("status")
        checks.append(QualityCheck(
            "reasoning_audit",
            reasoning_status == "pass",
            f"status={reasoning_status or 'missing'}",
        ))

    required_checks = [check for check in checks if check.required]
    passed_count = sum(1 for check in required_checks if check.passed)
    score = round(passed_count / len(required_checks), 4) if required_checks else 0.0
    return AnswerQualityScore(
        passed=all(check.passed for check in required_checks),
        score=score,
        checks=tuple(checks),
    )
