from __future__ import annotations

import json

from gesture_agent.core.models import SourceChunk
from gesture_agent.verification import GroundingVerifier


def _source(text: str, index: int = 1) -> SourceChunk:
    return SourceChunk(
        id=f"s{index}",
        title="旋钮",
        source="手册.docx",
        start_line=10,
        end_line=12,
        text=text,
    )


class FakeJudge:
    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts
        self.payload = None

    def chat(self, messages, **kwargs):
        self.payload = json.loads(messages[1]["content"])
        return json.dumps({
            "claims": [
                {
                    "id": claim["id"],
                    "verdict": self.verdicts.get(claim["id"], "supported"),
                    "confidence": 0.95,
                    "reason": "测试判定",
                }
                for claim in self.payload["claims"]
            ]
        }, ensure_ascii=False)


def test_supported_claim_passes_with_bound_evidence() -> None:
    judge = FakeJudge({"c1": "supported"})
    verifier = GroundingVerifier(judge)

    report = verifier.verify("旋钮可以承载角度属性。[1]", [_source("旋钮可以承载角度属性。")])

    assert report.status == "pass"
    assert report.score == 1.0
    assert report.claims[0].citation_ids == (1,)
    assert judge.payload["evidence"][0]["id"] == 1


def test_uncited_claim_is_rejected_without_calling_judge() -> None:
    verifier = GroundingVerifier(FakeJudge({}))

    report = verifier.verify("旋钮适合所有车载场景。", [_source("旋钮适合连续调节。")])

    assert report.should_retry
    assert report.claims[0].verdict == "unsupported"
    assert "没有绑定任何引用" in report.correction_hints


def test_partially_supported_claim_fails_in_strict_mode() -> None:
    verifier = GroundingVerifier(FakeJudge({"c1": "partially_supported"}), strict=True)

    report = verifier.verify("旋钮适合所有车载场景。[1]", [_source("旋钮可用于部分连续调节场景。")])

    assert report.status == "issues_found"
    assert report.should_retry
    assert report.score == 0.5


def test_citation_after_sentence_punctuation_stays_with_claim() -> None:
    verifier = GroundingVerifier(FakeJudge({"c1": "supported", "c2": "supported"}))
    report = verifier.verify(
        "旋钮可连续调节。[1] 它通过角度变化输入。[2]",
        [_source("连续调节。", 1), _source("通过角度变化输入。", 2)],
    )

    assert [claim.citation_ids for claim in report.claims] == [(1,), (2,)]


def test_evidence_limitation_statement_does_not_require_citation() -> None:
    verifier = GroundingVerifier(FakeJudge({}))
    report = verifier.verify("当前资料没有直接证据，无法确认该结论。", [_source("无关资料。")])

    assert report.status == "pass"
    assert report.claims[0].verdict == "supported"


def test_judge_failure_is_reported_as_unavailable() -> None:
    class BrokenJudge:
        def chat(self, messages, **kwargs):
            return "not-json"

    report = GroundingVerifier(BrokenJudge()).verify("旋钮可以连续调节。[1]", [_source("连续调节。")])

    assert report.status == "unavailable"
    assert report.error
