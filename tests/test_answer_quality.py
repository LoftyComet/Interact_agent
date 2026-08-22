from __future__ import annotations

from types import SimpleNamespace

from gesture_agent.core.models import QuestionStructure
from gesture_agent.verification import GroundingClaim, GroundingReport
from gesture_agent.verification.models import OutputVerificationResult
from web.backend.app import apply_grounding_safety_fallback, verify_answer_quality


class PassOutputVerifier:
    def verify(self, answer, structure, source_count=None):
        return OutputVerificationResult(status="pass")


class FailingGroundingVerifier:
    def verify(self, answer, chunks):
        claim = GroundingClaim(
            id="c1",
            section="直接回答",
            text="没有依据的结论。",
            verdict="unsupported",
            reason="引用资料未提及该结论",
        )
        return GroundingReport(
            status="issues_found",
            score=0.0,
            claims=(claim,),
            should_retry=True,
            correction_hints="删除没有依据的结论",
        )

    def sanitize(self, answer, report):
        safe_claim = GroundingClaim(
            id="c1", section="直接回答", text="有依据的结论。",
            citation_ids=(1,), verdict="supported", confidence=1.0, reason="资料支持",
        )
        return "有依据的结论。[1]", GroundingReport(
            status="pass", score=1.0, claims=(safe_claim,), should_retry=False,
        )


def test_quality_gate_combines_grounding_retry_and_issues() -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            verification=SimpleNamespace(verify_output=True, verify_grounding=True)
        ),
        output_verifier=PassOutputVerifier(),
        grounding_verifier=FailingGroundingVerifier(),
    )
    structure = QuestionStructure(
        raw_query="测试",
        intent="background_knowledge",
        layers=["background_knowledge"],
        terms=[],
        focus=["定义"],
    )

    result = verify_answer_quality(runtime, "没有依据的结论。", structure, [])

    assert result.should_retry
    assert "删除没有依据" in result.correction_hints
    assert result.grounding is not None
    assert any("引用资料未提及" in issue for issue in result.issue_descriptions)


def test_safety_fallback_replaces_failed_grounding_with_supported_subset() -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(verification=SimpleNamespace(verify_output=True)),
        output_verifier=PassOutputVerifier(),
        grounding_verifier=FailingGroundingVerifier(),
    )
    structure = QuestionStructure(
        raw_query="测试", intent="background_knowledge",
        layers=["background_knowledge"], terms=[], focus=["定义"],
    )
    failed = runtime.grounding_verifier.verify("没有依据的结论。", [])
    quality = SimpleNamespace(grounding=failed)

    answer, result = apply_grounding_safety_fallback(
        runtime, "没有依据的结论。", structure, [], quality
    )

    assert answer == "有依据的结论。[1]"
    assert result.grounding.status == "pass"
    assert result.issue_descriptions == []
