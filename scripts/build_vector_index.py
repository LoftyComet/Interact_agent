#!/usr/bin/env python3
"""Build or incrementally update chunk embeddings with SiliconFlow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing import ChunkVectorIndex, load_chunk_rows
from gesture_agent.providers.siliconflow import SiliconFlowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("knowledge_index/chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("knowledge_index/vectors"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    chunks = load_chunk_rows(args.chunks)
    client = SiliconFlowClient.from_env()
    existing = None
    if (args.output_dir / ChunkVectorIndex.META_FILE).exists():
        existing = ChunkVectorIndex.load(args.output_dir)
    index = ChunkVectorIndex.build(chunks, client, existing=existing, batch_size=args.batch_size)
    index.save(args.output_dir)
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "model": index.model,
                "count": len(index.rows),
                "dimension": int(index.vectors.shape[1]) if index.vectors.size else 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
