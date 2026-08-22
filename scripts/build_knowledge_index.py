#!/usr/bin/env python3
"""Build the complete deployable knowledge index in one command."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing import IndexBuildConfig, build_knowledge_index
from gesture_agent.providers.siliconflow import SiliconFlowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path, help="Root directory containing the source corpus")
    parser.add_argument("--output-dir", type=Path, default=Path("knowledge_index"))
    parser.add_argument("--term-inventory", type=Path, default=Path("data/term_inventory.json"))
    parser.add_argument("--structured-knowledge", type=Path, default=Path("data/structured_knowledge.json"))
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--vectors", action="store_true", help="Build embeddings using SILICONFLOW_API_KEY")
    parser.add_argument("--vector-batch-size", type=int, default=32)
    args = parser.parse_args()

    embedder = SiliconFlowClient.from_env() if args.vectors else None
    report = build_knowledge_index(
        IndexBuildConfig(
            corpus_root=args.corpus_root,
            output_dir=args.output_dir,
            term_inventory_path=args.term_inventory,
            structured_knowledge_path=args.structured_knowledge,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            vector_batch_size=args.vector_batch_size,
        ),
        embedder=embedder,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
