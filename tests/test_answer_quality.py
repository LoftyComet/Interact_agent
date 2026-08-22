from __future__ import annotations

from types import SimpleNamespace

from gesture_agent.core.models import QuestionStructure
from gesture_agent.verification import GroundingClaim, GroundingReport
from gesture_agent.verification.models import OutputVerificationResult
from web.backend.app import verify_answer_quality


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
