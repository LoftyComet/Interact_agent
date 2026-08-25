from __future__ import annotations

import json

from web.backend.app import create_app


def test_failure_collection_status_and_feedback_validation(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    config = tmp_path / "agent_config.json"
    config.write_text(
        json.dumps({
            "data": {
                "data_dir": "data",
                "term_inventory": "data/term_inventory.json",
                "structured_knowledge": "data/structured_knowledge.json",
                "output_frames": "data/output_frames.json",
                "knowledge_index": "knowledge_index",
            },
            "failure_collection": {
                "enabled": True,
                "database_path": str(database),
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    app = create_app(str(config))
    client = app.test_client()

    status = client.get("/api/failure_collection")
    missing = client.post(
        "/api/feedback",
        json={"response_id": "unknown", "rating": "down"},
    )
    invalid = client.post(
        "/api/feedback",
        json={"response_id": "unknown", "rating": "maybe"},
    )

    assert status.status_code == 200
    assert status.get_json()["enabled"] is True
    assert missing.status_code == 404
    assert invalid.status_code == 400


def test_negative_feedback_endpoint_promotes_recent_response(tmp_path) -> None:
    database = tmp_path / "candidates.sqlite"
    config = tmp_path / "agent_config.json"
    config.write_text(
        json.dumps({
            "data": {
                "data_dir": "data",
                "term_inventory": "data/term_inventory.json",
                "structured_knowledge": "data/structured_knowledge.json",
                "output_frames": "data/output_frames.json",
                "knowledge_index": "knowledge_index",
            },
            "failure_collection": {
                "enabled": True,
                "database_path": str(database),
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    app = create_app(str(config))
    collector = app.config["AGENT_RUNTIME"].failure_collector
    observed = collector.observe(
        {
            "response_id": "resp-feedback",
            "query": "什么是单击？",
            "answer": "待反馈回答",
        },
        triggers=[],
    )

    response = app.test_client().post(
        "/api/feedback",
        json={
            "response_id": observed.response_id,
            "rating": "down",
            "note": "编号不正确",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["candidate_id"].startswith("fail_")
