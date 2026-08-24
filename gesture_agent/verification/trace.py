from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def quality_snapshot(stage: str, answer: str, quality: Any) -> dict[str, Any]:
    """Serialize one verification attempt for evaluation-only observability."""

    grounding = getattr(quality, "grounding", None)
    reasoning = getattr(quality, "reasoning", None)
    return {
        "stage": stage,
        "answer": answer,
        "answer_length": len(answer),
        "should_retry": bool(getattr(quality, "should_retry", False)),
        "issues": list(getattr(quality, "issue_descriptions", []) or []),
        "grounding": grounding.to_dict() if grounding is not None else None,
        "reasoning": reasoning.to_dict() if reasoning is not None else None,
    }


def text_change_metrics(before: str, after: str) -> dict[str, Any]:
    """Measure material deletion separately from additions or rewrites."""

    deleted = 0
    added = 0
    for tag, before_start, before_end, after_start, after_end in SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag in {"delete", "replace"}:
            deleted += before_end - before_start
        if tag in {"insert", "replace"}:
            added += after_end - after_start
    return {
        "before_length": len(before),
        "after_length": len(after),
        "deleted_chars": deleted,
        "added_chars": added,
        "deletion_ratio": round(deleted / len(before), 4) if before else 0.0,
    }
