from scripts.evaluate_answer_quality import _summarize_verification_traces


def test_trace_summary_separates_first_pass_retry_and_fallback() -> None:
    results = [
        {"response": {"verification_trace": {
            "attempts": [{"should_retry": False}],
            "fallbacks": [],
        }}},
        {"response": {"verification_trace": {
            "attempts": [{"should_retry": True}, {"should_retry": False}],
            "fallbacks": [],
        }}},
        {"response": {"verification_trace": {
            "attempts": [{"should_retry": True}, {"should_retry": True}],
            "fallbacks": [{
                "kind": "grounding",
                "deletion_ratio": 0.25,
                "rejected_claims": [{"verdict": "partially_supported"}],
            }],
        }}},
    ]

    summary = _summarize_verification_traces(results)

    assert summary["first_pass_rate"] == 0.3333
    assert summary["retry_recovery_count"] == 1
    assert summary["fallback_kind_counts"] == {"grounding": 1}
    assert summary["rejected_claim_counts"] == {"partially_supported": 1}
    assert summary["mean_fallback_deletion_ratio"] == 0.25
