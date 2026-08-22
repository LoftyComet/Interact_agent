#!/usr/bin/env python3
"""Compile terminology, relations, and chunk-term mappings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing.taxonomy import build_taxonomy, match_chunk_terms, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term-inventory", type=Path, default=Path("data/term_inventory.json"))
    parser.add_argument("--structured-knowledge", type=Path, default=Path("data/structured_knowledge.json"))
    parser.add_argument("--chunks", type=Path, default=Path("knowledge_index/chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("knowledge_index"))
    args = parser.parse_args()

    terminology, relations = build_taxonomy(args.term_inventory, args.structured_knowledge)
    chunk_terms = match_chunk_terms(args.chunks, terminology)
    write_json(terminology, args.output_dir / "terminology.json")
    write_jsonl(relations, args.output_dir / "relations.jsonl")
    write_jsonl(chunk_terms, args.output_dir / "chunk_terms.jsonl")
    print(json.dumps({**terminology["summary"], "chunk_term_rows": len(chunk_terms)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
