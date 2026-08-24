from types import SimpleNamespace

from gesture_agent.verification import (
    GroundingClaim,
    GroundingReport,
    quality_snapshot,
    text_change_metrics,
)


def test_quality_snapshot_keeps_failed_claims_for_root_cause_analysis() -> None:
    report = GroundingReport(
        status="issues_found",
        score=0.5,
        claims=(GroundingClaim(
            id="c1",
            section="直接回答",
            text="没有依据的判断。",
            verdict="unsupported",
            reason="资料未提及",
        ),),
        should_retry=True,
    )
    quality = SimpleNamespace(
        grounding=report,
        reasoning=None,
        should_retry=True,
        issue_descriptions=["资料未提及"],
    )

    snapshot = quality_snapshot("initial", "没有依据的判断。", quality)

    assert snapshot["stage"] == "initial"
    assert snapshot["grounding"]["claims"][0]["verdict"] == "unsupported"
    assert snapshot["answer"] == "没有依据的判断。"


def test_text_change_metrics_reports_material_deletion() -> None:
    metrics = text_change_metrics("保留事实。删除猜测。", "保留事实。")

    assert metrics["deleted_chars"] == len("删除猜测。")
    assert metrics["added_chars"] == 0
    assert metrics["deletion_ratio"] == 0.5
