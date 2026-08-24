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


def test_line_ending_citation_applies_to_each_sentence_on_that_line() -> None:
    verifier = GroundingVerifier(FakeJudge({"c1": "supported", "c2": "supported"}))
    report = verifier.verify(
        "旋钮可以连续调节。它通过角度变化输入。[1]",
        [_source("旋钮可以连续调节，并通过角度变化输入。")],
    )

    assert [claim.citation_ids for claim in report.claims] == [(1,), (1,)]


def test_markdown_table_header_is_not_treated_as_claim() -> None:
    verifier = GroundingVerifier(FakeJudge({"c1": "supported"}))
    report = verifier.verify(
        "| 属性 | 说明 |\n|---|---|\n| 角度属性 | 连续旋转输入。[1] |",
        [_source("角度属性用于连续旋转输入。")],
    )

    assert len(report.claims) == 1
    assert report.claims[0].text == "角度属性 | 连续旋转输入。"


def test_list_lead_in_is_not_treated_as_standalone_claim() -> None:
    verifier = GroundingVerifier(FakeJudge({"c1": "supported"}))
    report = verifier.verify(
        "旋钮可以承载以下属性：\n- 角度属性用于连续旋转输入。[1]",
        [_source("角度属性用于连续旋转输入。")],
    )

    assert len(report.claims) == 1
    assert report.claims[0].text == "角度属性用于连续旋转输入。"


def test_evidence_limitation_statement_does_not_require_citation() -> None:
    verifier = GroundingVerifier(FakeJudge({}))
    report = verifier.verify("当前资料没有直接证据，无法确认该结论。", [_source("无关资料。")])

    assert report.status == "pass"
    assert report.claims[0].verdict == "supported"


def test_topic_prefixed_evidence_limitation_does_not_require_citation() -> None:
    verifier = GroundingVerifier(FakeJudge({}))
    answer = "至于是否影响实际使用，当前资料没有直接证据。\n词典资料未给出具体毫秒数值。"

    report = verifier.verify(answer, [_source("无关资料。")])

    assert report.status == "pass"
    assert all(claim.verdict == "supported" for claim in report.claims)


def test_claim_cannot_hide_unsupported_detail_behind_limitation_suffix() -> None:
    verifier = GroundingVerifier(FakeJudge({}))
    report = verifier.verify(
        "把菜单热区缩小到屏幕左上角（当前资料没有直接证据）。",
        [_source("边缘滑入通过空间分区避免冲突。")],
    )

    assert report.status == "issues_found"
    assert report.claims[0].verdict == "unsupported"


def test_judge_failure_is_reported_as_unavailable() -> None:
    class BrokenJudge:
        def chat(self, messages, **kwargs):
            return "not-json"

    report = GroundingVerifier(BrokenJudge()).verify("旋钮可以连续调节。[1]", [_source("连续调节。")])

    assert report.status == "unavailable"
    assert report.error


def test_malformed_judge_json_is_retried_once() -> None:
    class FlakyJsonJudge(FakeJudge):
        def __init__(self) -> None:
            super().__init__({})
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"claims":[{"id":"c1" "verdict":"supported"}]}'
            return json.dumps({
                "claims": [{
                    "id": "c1",
                    "verdict": "supported",
                    "confidence": 1.0,
                    "reason": "资料直接支持",
                }]
            }, ensure_ascii=False)

    judge = FlakyJsonJudge()
    report = GroundingVerifier(judge).verify(
        "旋钮可以连续调节。[1]",
        [_source("旋钮可以连续调节。")],
    )

    assert report.status == "pass"
    assert judge.calls == 2


def test_large_answer_is_judged_in_bounded_batches() -> None:
    class CountingJudge(FakeJudge):
        def __init__(self) -> None:
            super().__init__({})
            self.batch_sizes: list[int] = []

        def chat(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            self.batch_sizes.append(len(payload["claims"]))
            return json.dumps({
                "claims": [
                    {"id": claim["id"], "verdict": "supported", "confidence": 1, "reason": "有依据"}
                    for claim in payload["claims"]
                ]
            }, ensure_ascii=False)

    judge = CountingJudge()
    answer = "\n".join(f"旋钮属性事实陈述第{i}条。[1]" for i in range(1, 26))
    report = GroundingVerifier(judge).verify(answer, [_source("旋钮属性事实。")])

    assert report.status == "pass"
    assert len(report.claims) == 25
    assert judge.batch_sizes == [6, 6, 6, 6, 1]


def test_clarification_questions_are_not_treated_as_corpus_claims() -> None:
    verifier = GroundingVerifier(FakeJudge({}))
    answer = """资料边界有直接依据。[1]

## 需要澄清的信息

- 目标用户是谁？
- 是否需要盲操作？

## 可参考的机制

旋钮承载角度变化。[1]"""

    report = verifier.verify(answer, [_source("资料边界有直接依据；旋钮承载角度变化。")])

    assert report.status == "pass"
    assert [claim.section for claim in report.claims] == ["直接回答", "可参考的机制"]


def test_factual_statement_in_clarification_section_is_still_checked() -> None:
    verifier = GroundingVerifier(FakeJudge({}))
    report = verifier.verify(
        "## 需要澄清的信息\n\n是否需要盲操作？\n老人偏好大旋钮。",
        [_source("旋钮可以提供固定旋转轴。")],
    )

    assert report.status == "issues_found"
    assert [claim.text for claim in report.claims] == ["老人偏好大旋钮。"]


def test_sanitize_removes_only_claims_already_judged_unsafe() -> None:
    verifier = GroundingVerifier(FakeJudge({"c1": "supported", "c2": "partially_supported"}))
    answer = "旋钮承载角度属性。[1]\n旋钮适合所有场景。[1]"
    report = verifier.verify(answer, [_source("旋钮承载角度属性，仅适合部分场景。")])

    sanitized, safe_report = verifier.sanitize(answer, report)

    assert sanitized == "旋钮承载角度属性。[1]"
    assert safe_report.status == "pass"
    assert safe_report.score == 1.0
    assert [claim.text for claim in safe_report.claims] == ["旋钮承载角度属性。"]


def test_sanitize_iteratively_removes_fragments_created_by_partial_line_deletion() -> None:
    verifier = GroundingVerifier(FakeJudge({"c1": "supported", "c2": "partially_supported"}))
    answer = "旋钮承载角度属性。[1]；适合所有场景。[1]"
    report = verifier.verify(answer, [_source("旋钮承载角度属性。")])

    sanitized, safe_report = verifier.sanitize(answer, report)

    assert "适合所有场景" not in sanitized
    assert safe_report.status == "pass"
