from __future__ import annotations

import sqlite3

from gesture_agent.evaluation import FailureCollector
from scripts.review_failure_candidates import _load, _to_eval_draft


def test_approved_candidate_exports_as_non_authoritative_draft(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    collector = FailureCollector(database, enabled=True)
    observed = collector.observe(
        {
            "response_id": "resp-1",
            "query": "什么是单击？",
            "resolved_query": "请解释 IxDL 中的 1-b 单击",
            "structure": {"intent": "basic_interaction_mechanism", "subtype": ""},
            "provider": "deepseek",
            "model": "deepseek-chat",
            "answer": "错误回答",
            "answer_blocks": [],
            "chunks": [{"id": "chunk-1", "title": "单击", "citation": "book:1"}],
            "diagnostics": {"output_issues": ["缺少引用"]},
            "metadata": {},
        },
        triggers=["output_quality_failure"],
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE candidates SET status = 'approved' WHERE candidate_id = ?",
            (observed.candidate_id,),
        )

    row = _load(database, "approved", None)[0]
    draft = _to_eval_draft(row)

    assert draft["query"] == "什么是单击？"
    assert draft["resolved_query"] == "请解释 IxDL 中的 1-b 单击"
    assert draft["observed_failure"]["chunk_refs"][0]["id"] == "chunk-1"
    assert draft["expected_output"]["criteria_are_authoritative_gold_facts"] is False
    assert draft["expected_output"]["criteria_status"] == "needs_human_review"
