#!/usr/bin/env python3
"""Build the deployable SQLite metadata and BM25 index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing import LexicalIndex, load_chunk_rows, load_chunk_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("knowledge_index/chunks.jsonl"))
    parser.add_argument("--chunk-terms", type=Path, default=Path("knowledge_index/chunk_terms.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("knowledge_index/knowledge.sqlite"))
    args = parser.parse_args()

    chunks = load_chunk_rows(args.chunks)
    terms = load_chunk_terms(args.chunk_terms)
    index = LexicalIndex.build(args.output, chunks, terms)
    print(json.dumps({"output": str(index.path.resolve()), "chunk_count": len(chunks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
