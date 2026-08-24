from gesture_agent.evaluation import score_api_response
from gesture_agent.knowledge import MechanismRegistry


def _case(mode="corpus_plus_labeled_reasoning"):
    return {
        "intent": {
            "expected_runtime_label": "interaction_compare",
            "expected_subtype": "control_form_compare",
        },
        "response_contract": {
            "reasoning_allowed": mode != "corpus_only",
            "required_ui_blocks": ["corpus_evidence"],
            "allowed_ui_blocks": ["corpus_evidence", "design_reasoning"],
        },
    }


def _response():
    return {
        "status": "ready",
        "answer": "1-b 单击（Single Tap）是离散触发。[1]",
        "answer_blocks": [{
            "type": "corpus_evidence",
            "title": "语料依据",
            "markdown": "1-b 单击（Single Tap）是离散触发。[1]",
            "citations": [1],
        }],
        "structure": {
            "intent": "interaction_compare",
            "subtype": "control_form_compare",
        },
        "chunks": [{"id": "chunk-1"}],
        "output_issues": [],
        "safety_fallback_applied": False,
        "grounding": {"status": "pass", "score": 1.0},
        "reasoning_audit": None,
    }


def test_evaluator_passes_complete_hard_contract() -> None:
    score = score_api_response(
        _case(),
        _response(),
        MechanismRegistry.load("data/term_inventory.json"),
    )

    assert score.passed
    assert score.score == 1.0


def test_evaluator_fails_invalid_citation_and_mechanism_code() -> None:
    response = _response()
    response["answer"] = "2-a 单击是离散触发。[3]"
    response["answer_blocks"][0]["markdown"] = response["answer"]
    response["answer_blocks"][0]["citations"] = [3]

    score = score_api_response(
        _case(),
        response,
        MechanismRegistry.load("data/term_inventory.json"),
    )
    failed = {check.name for check in score.checks if not check.passed}

    assert "citation_integrity" in failed
    assert "mechanism_registry" in failed


def test_evaluator_flags_last_resort_claim_deletion() -> None:
    response = _response()
    response["safety_fallback_applied"] = True

    score = score_api_response(
        _case(),
        response,
        MechanismRegistry.load("data/term_inventory.json"),
    )

    assert any(
        check.name == "no_safety_deletion" and not check.passed
        for check in score.checks
    )


def test_evaluator_requires_audit_for_reasoning_block() -> None:
    response = _response()
    response["answer_blocks"].append({
        "type": "design_reasoning",
        "title": "设计推导（仅供参考）",
        "markdown": "可以尝试另一个方案。",
        "citations": [],
    })

    score = score_api_response(
        _case(),
        response,
        MechanismRegistry.load("data/term_inventory.json"),
    )

    assert any(
        check.name == "reasoning_audit" and not check.passed
        for check in score.checks
    )


def test_evaluator_accepts_honest_limitation_without_fake_citation() -> None:
    response = _response()
    response["answer"] = "当前资料没有直接证据支持更具体的结论。"
    response["answer_blocks"][0]["markdown"] = response["answer"]
    response["answer_blocks"][0]["citations"] = []

    score = score_api_response(
        _case(),
        response,
        MechanismRegistry.load("data/term_inventory.json"),
    )

    assert next(
        check for check in score.checks if check.name == "corpus_citation_present"
    ).passed
