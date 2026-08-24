import json

from gesture_agent.core.models import SourceChunk
from gesture_agent.verification import ReasoningVerifier


class FakeJudge:
    def __init__(self, issues):
        self.issues = issues
        self.payload = None

    def chat(self, messages, **kwargs):
        self.payload = json.loads(messages[1]["content"])
        return json.dumps({"issues": self.issues}, ensure_ascii=False)


def _source():
    return SourceChunk(
        id="s1", title="旋钮", source="book.docx",
        start_line=1, end_line=2, text="旋钮可以提供固定旋转轴。",
    )


def test_reasoning_hypothesis_passes_when_auditor_finds_no_policy_issue() -> None:
    judge = FakeJudge([])
    report = ReasoningVerifier(judge).verify(
        "可以尝试用旋钮承担主要调节任务，但需要用户测试验证。",
        [_source()],
    )

    assert report.status == "pass"
    assert judge.payload["evidence"][0]["id"] == 1


def test_external_parameter_requires_retry() -> None:
    report = ReasoningVerifier(FakeJudge([{
        "issue_type": "external_fact",
        "text": "热区设置为 20pt",
        "reason": "资料没有该参数",
    }])).verify("可以把热区设为 20pt。", [_source()])

    assert report.status == "issues_found"
    assert report.should_retry
    assert "external_fact" in report.correction_hints


def test_missing_judge_is_reported_without_claiming_pass() -> None:
    report = ReasoningVerifier(None).verify("可以尝试新方案。", [_source()])

    assert report.status == "unavailable"


def test_deterministic_audit_rejects_unsourced_parameter_and_absolute_claim() -> None:
    report = ReasoningVerifier(FakeJudge([])).verify(
        "把热区设置为 20pt。这样可发现性会显著提升。",
        [_source()],
    )

    assert report.status == "issues_found"
    assert {issue.issue_type for issue in report.issues} == {
        "external_fact", "unlabeled_assertion"
    }
