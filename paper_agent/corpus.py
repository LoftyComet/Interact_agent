"""语料存储:corpus.jsonl 的读写与去重。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .models import Paper


def write_corpus(path: str | Path, papers: Iterable[Paper], *, mode: str = "w") -> int:
    """把论文写入 JSONL,每行一个 Paper。返回写入条数。

    mode="a" 时追加(用于增量抓取)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(mode, encoding="utf-8") as fh:
        for paper in papers:
            fh.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_corpus(path: str | Path) -> list[Paper]:
    """从 JSONL 读取语料,自动去重。文件不存在时返回空列表。"""
    path = Path(path)
    if not path.exists():
        return []
    papers: list[Paper] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        data = json.loads(line)
        papers.append(_paper_from_dict(data))
    return dedupe(papers)


def dedupe(papers: list[Paper]) -> list[Paper]:
    """按 id 去重,回退到 doi / 小写标题。保留首次出现的记录。"""
    seen: set[str] = set()
    result: list[Paper] = []
    for paper in papers:
        key = paper.id or paper.doi or paper.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(paper)
    return result


def _paper_from_dict(data: dict) -> Paper:
    """从 dict 构造 Paper,忽略未知字段,补齐缺省值。"""
    return Paper(
        id=data.get("id", ""),
        title=data.get("title", ""),
        authors=list(data.get("authors", []) or []),
        year=data.get("year"),
        venue=data.get("venue", ""),
        abstract=data.get("abstract", ""),
        doi=data.get("doi", "") or "",
        url=data.get("url", "") or "",
        citations=int(data.get("citations", 0) or 0),
        concepts=list(data.get("concepts", []) or []),
    )
