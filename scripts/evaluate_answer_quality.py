#!/usr/bin/env python3
"""Run the versioned IxDL answer set through the local API and score hard checks."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.evaluation import score_api_response
from gesture_agent.knowledge import MechanismRegistry
from web.backend.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/ixdl_answer_quality_v1.jsonl"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/term_inventory.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/ixdl_answer_quality_latest.json"),
    )
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--style", default="concise", choices=["concise", "detailed"])
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--include-feedback", action="store_true")
    args = parser.parse_args()

    cases = _load_jsonl(args.dataset)
    selected = [case for case in cases if _select_case(case, args.ids, args.include_feedback)]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]

    app = create_app()
    app.testing = True
    client = app.test_client()
    registry = MechanismRegistry.load(args.registry)
    results: list[dict[str, Any]] = []

    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case['id']} {case['query']}", flush=True)
        response = client.post("/api/ask", json={
            "question": case["query"],
            "session_id": f"eval-{case['id']}-{uuid.uuid4().hex}",
            "provider": args.provider,
            "style": args.style,
        })
        payload = response.get_json(silent=True) or {"status": "http_error"}
        score = score_api_response(case, payload, registry)
        results.append({
            "id": case["id"],
            "query": case["query"],
            "http_status": response.status_code,
            "score": score.to_dict(),
            "response": payload,
        })
        failed = [check.name for check in score.checks if check.required and not check.passed]
        warnings = [check.name for check in score.checks if not check.required and not check.passed]
        print(
            f"  score={score.score:.2%} pass={score.passed} failed={failed} warnings={warnings}",
            flush=True,
        )

    passed = sum(1 for result in results if result["score"]["passed"])
    fallback_count = sum(
        1 for result in results
        if result["response"].get("safety_fallback_applied")
    )
    mean_score = (
        round(sum(result["score"]["score"] for result in results) / len(results), 4)
        if results else 0.0
    )
    summary = {
        "case_count": len(results),
        "passed": passed,
        "pass_rate": _ratio(passed, len(results)),
        "mean_hard_check_score": mean_score,
        "safety_fallback_count": fallback_count,
        "safety_fallback_rate": _ratio(fallback_count, len(results)),
        "provider": args.provider,
        "style": args.style,
    }
    output = {
        "schema_version": "1.0",
        "dataset": str(args.dataset),
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **summary}, ensure_ascii=False, indent=2))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_case(case: dict[str, Any], ids: list[str], include_feedback: bool) -> bool:
    if ids and case.get("id") not in ids:
        return False
    contract = case.get("response_contract", {})
    if contract.get("requires_prior_context_fixture") and not include_feedback:
        return False
    return bool(case.get("intent", {}).get("evaluable", True))


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


if __name__ == "__main__":
    main()
