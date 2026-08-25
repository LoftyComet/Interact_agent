#!/usr/bin/env python3
"""Review and manually promote runtime failure candidates into an eval draft."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


JSON_COLUMNS = {
    "triggers_json": "triggers",
    "answer_blocks_json": "answer_blocks",
    "chunk_refs_json": "chunk_refs",
    "diagnostics_json": "diagnostics",
    "metadata_json": "metadata",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("runtime/evaluation_candidates.sqlite"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List candidate summaries")
    list_parser.add_argument("--status", default="pending")
    list_parser.add_argument("--limit", type=int, default=30)

    export_parser = subparsers.add_parser("export", help="Export candidates for review")
    export_parser.add_argument("--status", default="pending")
    export_parser.add_argument("--output", type=Path, required=True)

    status_parser = subparsers.add_parser("set-status", help="Apply a human review decision")
    status_parser.add_argument(
        "--status",
        required=True,
        choices=["pending", "approved", "rejected", "resolved"],
    )
    status_parser.add_argument("--ids", nargs="+", required=True)

    promote_parser = subparsers.add_parser(
        "promote",
        help="Export approved candidates as a non-authoritative evaluation draft",
    )
    promote_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if not args.database.exists():
        raise SystemExit(f"Candidate database does not exist: {args.database}")

    if args.command == "list":
        rows = _load(args.database, args.status, args.limit)
        for row in rows:
            print(
                f"{row['candidate_id']} count={row['occurrence_count']} "
                f"intent={row['intent'] or '-'} triggers={','.join(row['triggers'])}\n"
                f"  {row['query_text'] or '[query hidden]'}"
            )
        print(f"count={len(rows)}")
    elif args.command == "export":
        rows = _load(args.database, args.status, None)
        _write_jsonl(args.output, rows)
        print(json.dumps({"output": str(args.output), "count": len(rows)}, ensure_ascii=False))
    elif args.command == "set-status":
        with sqlite3.connect(args.database) as connection:
            placeholders = ",".join("?" for _ in args.ids)
            cursor = connection.execute(
                f"UPDATE candidates SET status = ? WHERE candidate_id IN ({placeholders})",
                [args.status, *args.ids],
            )
        print(json.dumps({"updated": cursor.rowcount, "status": args.status}, ensure_ascii=False))
    elif args.command == "promote":
        rows = _load(args.database, "approved", None)
        drafts = [_to_eval_draft(row) for row in rows]
        _write_jsonl(args.output, drafts)
        print(json.dumps({"output": str(args.output), "count": len(drafts)}, ensure_ascii=False))


def _load(database: Path, status: str, limit: int | None) -> list[dict[str, Any]]:
    query = "SELECT * FROM candidates WHERE status = ? ORDER BY occurrence_count DESC, last_seen DESC"
    params: list[Any] = [status]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(query, params)]
    for row in rows:
        for source, target in JSON_COLUMNS.items():
            row[target] = json.loads(row.pop(source))
    return rows


def _to_eval_draft(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": f"runtime-{row['candidate_id']}",
        "source": {
            "type": "runtime_failure_candidate",
            "candidate_id": row["candidate_id"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "occurrence_count": row["occurrence_count"],
        },
        "query": row["query_text"],
        "resolved_query": row["metadata"].get("resolved_query") or row["query_text"],
        "intent": {
            "expected_runtime_label": row["intent"] or None,
            "expected_subtype": row["subtype"] or None,
            "requires_human_confirmation": True,
        },
        "observed_failure": {
            "triggers": row["triggers"],
            "diagnostics": row["diagnostics"],
            "answer": row["answer_text"],
            "answer_blocks": row["answer_blocks"],
            "chunk_refs": row["chunk_refs"],
            "user_rating": row["user_rating"],
            "user_note": row["user_note"],
        },
        "expected_output": {
            "criteria_status": "needs_human_review",
            "criteria_are_authoritative_gold_facts": False,
            "content_criteria": [],
        },
        "dataset_status": "approved_candidate_pending_labeling",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
