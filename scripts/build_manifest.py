#!/usr/bin/env python3
"""Create manifest.json for a deployable knowledge-index directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_agent.indexing import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path, nargs="?", default=Path("knowledge_index"))
    args = parser.parse_args()
    manifest = build_manifest(args.index_dir)
    output = write_manifest(manifest, args.index_dir / "manifest.json")
    print(json.dumps({"output": str(output), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
