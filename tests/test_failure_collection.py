from __future__ import annotations

import json
import sqlite3

from gesture_agent.evaluation import FailureCollector, detect_failure_signals


def _record(response_id: str = "resp-1") -> dict:
    return {
        "response_id": response_id,
        "query": "联系 me@example.com，为什么 1-b 单击会失败？",
        "structure": {"intent": "basic_interaction_mechanism", "subtype": ""},
        "provider": "deepseek",
        "model": "deepseek-chat",
        "answer": "请联系 me@example.com。",
        "answer_blocks": [{"type": "corpus_evidence", "markdown": "回答"}],
        "chunks": [{
            "id": "chunk-1",
            "title": "单击",
            "citation": "book.docx:10-12",
            "source": "book.docx",
            "start_line": 10,
            "end_line": 12,
            "score": 0.9,
            "text": "原始语料正文不应写入失败库",
        }],
        "diagnostics": {"output_issues": ["编号错误"], "retry_count": 1},
        "metadata": {"endpoint": "/api/ask"},
    }


def test_failure_collector_persists_redacted_deduplicated_candidates(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    collector = FailureCollector(database, enabled=True)

    first = collector.observe(_record(), triggers=["generation_retry"])
    second = collector.observe(_record("resp-2"), triggers=["generation_retry"])

    assert first.candidate_id == second.candidate_id
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT occurrence_count, query_text, answer_text, chunk_refs_json FROM candidates"
        ).fetchone()
    assert row[0] == 2
    assert "me@example.com" not in row[1]
    assert "[REDACTED_EMAIL]" in row[1]
    assert "me@example.com" not in row[2]
    assert "原始语料正文" not in row[3]
    assert json.loads(row[3])[0]["id"] == "chunk-1"


def test_negative_feedback_promotes_recent_success_without_storing_all_successes(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    collector = FailureCollector(database, enabled=True)
    observed = collector.observe(_record(), triggers=[])

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM candidates").fetchone()[0] == 0

    candidate_id = collector.record_feedback(
        observed.response_id,
        rating="down",
        note="引用内容不对",
    )

    assert candidate_id
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT user_rating, user_note, triggers_json FROM candidates"
        ).fetchone()
    assert row[0:2] == ("down", "引用内容不对")
    assert json.loads(row[2]) == ["user_negative"]


def test_negative_feedback_enriches_existing_candidate_without_duplication(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    collector = FailureCollector(database, enabled=True)
    observed = collector.observe(_record(), triggers=["generation_retry"])

    candidate_id = collector.record_feedback(
        observed.response_id,
        rating="down",
        note="回答仍然不正确",
    )

    assert candidate_id == observed.candidate_id
    with sqlite3.connect(database) as connection:
        count = connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
        triggers = json.loads(
            connection.execute("SELECT triggers_json FROM candidates").fetchone()[0]
        )
    assert count == 1
    assert triggers == ["generation_retry", "user_negative"]


def test_dedup_merges_failure_modes_and_keeps_every_response_feedback_link(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    collector = FailureCollector(database, enabled=True)
    first = collector.observe(_record("resp-1"), triggers=["generation_retry"])
    second = collector.observe(_record("resp-2"), triggers=["safety_fallback"])

    feedback_candidate = collector.record_feedback(
        first.response_id,
        rating="down",
        note="第一条回答的引用有误",
    )

    assert first.candidate_id == second.candidate_id == feedback_candidate
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT occurrence_count, triggers_json, user_note FROM candidates"
        ).fetchone()
    assert row[0] == 2
    assert json.loads(row[1]) == [
        "generation_retry",
        "safety_fallback",
        "user_negative",
    ]
    assert row[2] == "第一条回答的引用有误"


def test_failure_signals_cover_retries_fallbacks_and_audit_failures() -> None:
    signals = detect_failure_signals(
        output_issues=["编号错误"],
        grounding_status="issues_found",
        reasoning_status="pass",
        safety_fallback_applied=True,
        retry_count=2,
    )

    assert signals == [
        "output_quality_failure",
        "grounding_issues_found",
        "safety_fallback",
        "generation_retry",
    ]


def test_query_storage_can_be_disabled_for_privacy(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    collector = FailureCollector(database, enabled=True, store_raw_query=False)
    collector.observe(
        {**_record(), "resolved_query": "包含用户上下文的解析后问题"},
        triggers=["generation_retry"],
    )

    with sqlite3.connect(database) as connection:
        query_text, query_hash, metadata_json = connection.execute(
            "SELECT query_text, query_sha256, metadata_json FROM candidates"
        ).fetchone()
    assert query_text == ""
    assert len(query_hash) == 64
    assert json.loads(metadata_json)["resolved_query"] == ""
