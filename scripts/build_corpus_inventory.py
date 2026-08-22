#!/usr/bin/env python3
"""Build a versioned inventory for a local corpus directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing import build_inventory, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path, help="Directory containing source documents")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("knowledge_index/corpus_inventory.json"),
        help="Inventory JSON output path",
    )
    args = parser.parse_args()

    inventory = build_inventory(args.corpus_root)
    output = write_inventory(inventory, args.output)
    print(json.dumps({"output": str(output), **inventory.summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
