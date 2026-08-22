#!/usr/bin/env python3
"""Build structure-aware retrieval chunks from normalized documents JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing import chunk_documents, load_documents, write_build_report, write_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("knowledge_index"))
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args()

    documents = load_documents(args.documents)
    chunks, report = chunk_documents(documents, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    chunks_path = write_chunks(chunks, args.output_dir / "chunks.jsonl")
    report_path = write_build_report(report, args.output_dir / "chunk_report.json")
    print(json.dumps({"chunks": str(chunks_path), "report": str(report_path), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
