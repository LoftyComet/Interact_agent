from __future__ import annotations

from types import SimpleNamespace

from gesture_agent.core.models import QuestionStructure
from gesture_agent.knowledge import MechanismRegistry
from gesture_agent.verification import (
    GroundingClaim,
    GroundingReport,
    ReasoningIssue,
    ReasoningReport,
    ReasoningVerifier,
    parse_answer_blocks,
)
from gesture_agent.verification.models import OutputIssue, OutputVerificationResult
from gesture_agent.core.models import IntentOutputFrames
from web.backend.app import (
    apply_grounding_safety_fallback,
    apply_reasoning_safety_fallback,
    generation_temperature,
    normalize_mechanism_codes,
    repair_unlabeled_design_reasoning,
    refresh_output_quality,
    repair_safe_answer_contract,
    verify_answer_quality,
)


class PassOutputVerifier:
    def verify(self, answer, structure, source_count=None):
        return OutputVerificationResult(status="pass")


class RequirePressCodeOutputVerifier:
    def verify(self, answer, structure, source_count=None):
        if "1-c 按下" in answer:
            return OutputVerificationResult(status="pass")
        issue = OutputIssue(
            issue_type="missing_mechanism_code",
            location="按下",
            description="按下缺少编号",
            severity="error",
        )
        return OutputVerificationResult(
            status="issues_found",
            issues=(issue,),
            should_retry=True,
            correction_hints="补上编号",
        )


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


def test_reasoning_audit_noop_does_not_count_as_safety_fallback() -> None:
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
当前资料没有直接证据支持该方案。
<!-- ixdl-answer-block:design_reasoning -->
可以尝试把这个方案作为待验证假设。"""
    report = ReasoningReport(
        status="issues_found",
        issues=(ReasoningIssue(
            issue_type="unlabeled_assertion",
            text="可以尝试把这个方案作为待验证假设。",
            reason="模型误报",
        ),),
        should_retry=True,
    )
    quality = SimpleNamespace(
        grounding=GroundingReport(status="pass", score=1.0),
        reasoning=report,
        safety_fallback_applied=False,
    )

    sanitized, result = apply_reasoning_safety_fallback(
        runtime, answer, structure, [], quality
    )

    assert sanitized == answer
    assert result.reasoning.status == "pass"
    assert result.safety_fallback_applied is False


def test_post_fallback_normalization_refreshes_stale_output_issues() -> None:
    runtime = SimpleNamespace(
        kb=SimpleNamespace(
            mechanism_registry=MechanismRegistry.load("data/term_inventory.json")
        ),
        config=SimpleNamespace(verification=SimpleNamespace(verify_output=True)),
        output_verifier=RequirePressCodeOutputVerifier(),
    )
    structure = QuestionStructure(
        raw_query="按下是什么", intent="basic_interaction_mechanism",
        layers=["interaction_mechanism"], terms=["按下"], focus=["定义"],
    )
    stale = SimpleNamespace(
        grounding=GroundingReport(status="pass", score=1.0),
        reasoning=None,
        safety_fallback_applied=True,
    )

    answer = normalize_mechanism_codes(runtime, "按下是一种交互机制。", structure)
    refreshed = refresh_output_quality(runtime, answer, structure, [], stale)

    assert answer == "1-c 按下是一种交互机制。"
    assert refreshed.issue_descriptions == []
    assert refreshed.should_retry is False


def test_unlabeled_design_assessment_is_moved_out_of_corpus_block() -> None:
    structure = QuestionStructure(
        raw_query="长按对焦松手拍照是否可行",
        intent="design_evaluation",
        layers=["design_evaluation"],
        terms=["长按"],
        focus=["设计评估"],
        reasoning_allowed=True,
    )
    answer = """<!-- ixdl-answer-block:corpus_evidence -->
这个方案可行但有风险。

## 问题诊断

松手触发可能误操作。

## 修改建议

建议增加状态反馈。"""

    repaired = repair_unlabeled_design_reasoning(answer, structure)

    assert "当前资料没有直接证据支持" in repaired
    assert "<!-- ixdl-answer-block:design_reasoning -->" in repaired
    assert repaired.index("design_reasoning") < repaired.index("这个方案可行但有风险")


def test_direct_book_guidance_without_design_markers_stays_in_corpus() -> None:
    structure = QuestionStructure(
        raw_query="这个方案如何",
        intent="design_evaluation",
        layers=["design_evaluation"],
        terms=[],
        focus=["设计评估"],
        reasoning_allowed=True,
    )
    answer = "<!-- ixdl-answer-block:corpus_evidence -->\n书中案例直接采用了这一结构。[1]"

    assert repair_unlabeled_design_reasoning(answer, structure) == answer


def test_product_mapping_leaked_before_existing_reasoning_marker_is_moved() -> None:
    structure = QuestionStructure(
        raw_query="我把播放器的进度条理解成拖拽",
        intent="function_interaction_breakdown",
        layers=["interaction_mechanism"],
        terms=["拖拽"],
        focus=["逻辑关系"],
        reasoning_allowed=True,
    )
    answer = """<!-- ixdl-answer-block:corpus_evidence -->
按你的描述，播放器进度条可能对应2-a 拖拽。

## 功能概述

你拆的是视频播放器功能。

<!-- ixdl-answer-block:design_reasoning -->
可以尝试继续验证音量映射。"""

    repaired = repair_unlabeled_design_reasoning(answer, structure)
    document = parse_answer_blocks(repaired)

    assert "按你的描述" not in document.corpus_markdown
    assert document.corpus_markdown.startswith("当前资料没有直接证据")
    assert "按你的描述" in document.markdown_for("design_reasoning")
    assert "继续验证音量映射" in document.markdown_for("design_reasoning")


def test_corpus_only_generation_uses_near_deterministic_temperature() -> None:
    runtime = SimpleNamespace(config=SimpleNamespace(temperature=0.2))
    factual = QuestionStructure(
        raw_query="范畴论在书里有什么用",
        intent="background_knowledge",
        layers=["background_knowledge"],
        terms=[],
        focus=["定义"],
        reasoning_allowed=False,
    )
    design = QuestionStructure(
        raw_query="帮我设计方案",
        intent="design_suggestion",
        layers=["design_evaluation"],
        terms=[],
        focus=["设计建议"],
        reasoning_allowed=True,
    )

    assert generation_temperature(runtime, factual) == 0.05
    assert generation_temperature(runtime, design) == 0.2
