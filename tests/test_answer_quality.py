from __future__ import annotations

from types import SimpleNamespace

from gesture_agent.core.models import QuestionStructure
from gesture_agent.verification import (
    GroundingClaim,
    GroundingReport,
    ReasoningIssue,
    ReasoningReport,
    ReasoningVerifier,
)
from gesture_agent.verification.models import OutputVerificationResult
from gesture_agent.core.models import IntentOutputFrames
from web.backend.app import (
    apply_grounding_safety_fallback,
    apply_reasoning_safety_fallback,
    repair_safe_answer_contract,
    verify_answer_quality,
)


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


class CapturingGroundingVerifier:
    def __init__(self):
        self.verified_answer = ""

    def verify(self, answer, chunks):
        self.verified_answer = answer
        return GroundingReport(status="pass", score=1.0, should_retry=False)


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
    assert result.safety_fallback_applied is True


def test_grounding_checks_only_corpus_evidence_block() -> None:
    grounding_verifier = CapturingGroundingVerifier()
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            verification=SimpleNamespace(verify_output=False, verify_grounding=True)
        ),
        grounding_verifier=grounding_verifier,
    )
    structure = QuestionStructure(
        raw_query="测试", intent="design_suggestion",
        layers=["design_evaluation"], terms=[], focus=["设计评估"], reasoning_allowed=True,
    )
    answer = """<!-- ixdl-answer-block:corpus_evidence -->
语料结论。[1]
<!-- ixdl-answer-block:design_reasoning -->
可以尝试一个待验证的方案。"""

    result = verify_answer_quality(runtime, answer, structure, [])

    assert result.should_retry is False
    assert grounding_verifier.verified_answer == "语料结论。[1]"
    assert "待验证" not in grounding_verifier.verified_answer


def test_safety_fallback_preserves_labeled_reasoning_block() -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(verification=SimpleNamespace(verify_output=False)),
        grounding_verifier=FailingGroundingVerifier(),
    )
    structure = QuestionStructure(
        raw_query="测试", intent="design_suggestion",
        layers=["design_evaluation"], terms=[], focus=["设计评估"], reasoning_allowed=True,
    )
    answer = """<!-- ixdl-answer-block:corpus_evidence -->
没有依据的结论。
<!-- ixdl-answer-block:design_reasoning -->
可以尝试一个待验证的方案。"""
    failed = runtime.grounding_verifier.verify("没有依据的结论。", [])
    quality = SimpleNamespace(grounding=failed)

    sanitized, result = apply_grounding_safety_fallback(runtime, answer, structure, [], quality)

    assert "有依据的结论。[1]" in sanitized
    assert "没有依据的结论" not in sanitized
    assert "可以尝试一个待验证的方案" in sanitized
    assert result.grounding.status == "pass"


def test_safe_contract_repair_fills_deleted_direct_answer_and_empty_section() -> None:
    structure = QuestionStructure(
        raw_query="测试", intent="interaction_optimization",
        layers=["design_evaluation"], terms=[], focus=["设计评估"],
    )
    frames = IntentOutputFrames(frames={
        "interaction_optimization": ["现状复述", "优化建议"],
    })
    broken = """<!-- ixdl-answer-block:corpus_evidence -->
## 现状复述
有依据的诊断。[1]
## 优化建议
以下建议供参考："""

    repaired = repair_safe_answer_contract(broken, structure, frames)

    assert "当前资料不足以支持更具体的结论" in repaired
    assert "## 优化建议\n\n当前资料没有直接证据" in repaired
    assert "有依据的诊断。[1]" in repaired


def test_safe_contract_repair_does_not_duplicate_section_from_reasoning_block() -> None:
    structure = QuestionStructure(
        raw_query="测试", intent="design_suggestion",
        layers=["design_evaluation"], terms=[], focus=["设计建议"], reasoning_allowed=True,
    )
    frames = IntentOutputFrames(frames={
        "design_suggestion": ["需求理解", "初步建议"],
    })
    broken = """<!-- ixdl-answer-block:corpus_evidence -->
有依据的结论。[1]
## 需求理解
有依据的需求说明。[1]
<!-- ixdl-answer-block:design_reasoning -->
## 初步建议
可以尝试待验证方案。"""

    repaired = repair_safe_answer_contract(broken, structure, frames)

    assert repaired.count("## 初步建议") == 1
    assert "可以尝试待验证方案" in repaired


def test_reasoning_fallback_preserves_safe_ideas_instead_of_dropping_block() -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(verification=SimpleNamespace(verify_output=True)),
        output_verifier=PassOutputVerifier(),
        reasoning_verifier=ReasoningVerifier(None),
    )
    structure = QuestionStructure(
        raw_query="测试", intent="design_suggestion",
        layers=["design_evaluation"], terms=[], focus=["设计建议"], reasoning_allowed=True,
    )
    answer = """<!-- ixdl-answer-block:corpus_evidence -->
资料边界。[1]
<!-- ixdl-answer-block:design_reasoning -->
- 把热区设为 20pt。
- 可以尝试保留按钮作为替代入口。"""
    report = ReasoningReport(
        status="issues_found",
        issues=(ReasoningIssue(
            issue_type="external_fact",
            text="20pt",
            reason="语料没有该参数",
        ),),
        should_retry=True,
    )
    quality = SimpleNamespace(grounding=None, reasoning=report)

    sanitized, result = apply_reasoning_safety_fallback(
        runtime, answer, structure, [], quality
    )

    assert "20pt" not in sanitized
    assert "可以尝试保留按钮作为替代入口" in sanitized
    assert "ixdl-answer-block:design_reasoning" in sanitized
    assert result.reasoning.status == "pass"
    assert result.safety_fallback_applied is True
