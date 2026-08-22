#!/usr/bin/env python3
"""Extract included inventory files into format-neutral JSONL documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing import extract_inventory, load_inventory, write_build_report, write_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("knowledge_index"))
    args = parser.parse_args()

    inventory = load_inventory(args.inventory)
    documents, report = extract_inventory(inventory)
    documents_path = write_documents(documents, args.output_dir / "documents.jsonl")
    report_path = write_build_report(report, args.output_dir / "extraction_report.json")
    print(json.dumps({"documents": str(documents_path), "report": str(report_path), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
