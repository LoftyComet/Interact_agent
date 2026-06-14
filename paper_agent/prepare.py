"""数据准备:抓取语料 + 构建向量索引。

由于不做交互式 CLI,这里把 fetch / embed 封装成函数,供 web 后端的
后台管理端点调用,也可在脚本中直接调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from gesture_agent.providers.siliconflow import SiliconFlowClient
from .config import PaperAgentConfig
from .corpus import dedupe, load_corpus, write_corpus
from .index import VectorIndex
from .models import Paper
from .sources import SemanticScholarClient

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def fetch_corpus(config: PaperAgentConfig, *, progress: Optional[ProgressFn] = None) -> int:
    """按配置抓取论文,合并进已有语料,写回 corpus.jsonl。返回语料总条数。"""
    progress = progress or _noop
    client = SemanticScholarClient.from_env()
    existing = load_corpus(config.corpus_path)
    progress(f"已有语料 {len(existing)} 篇,开始抓取 venues={config.venues} "
             f"years={config.from_year}-{config.to_year}")

    harvested: list[Paper] = []
    for venue in config.venues:
        before = len(harvested)
        for paper in client.fetch([venue], config.years):
            if config.require_abstract and not paper.abstract:
                continue
            harvested.append(paper)
        progress(f"  {venue}: +{len(harvested) - before} 篇")

    merged = dedupe(existing + harvested)
    write_corpus(config.corpus_path, merged, mode="w")
    progress(f"语料合并后共 {len(merged)} 篇,已写入 {config.corpus_path}")
    return len(merged)


def build_index(
    config: PaperAgentConfig,
    *,
    rebuild: bool = False,
    progress: Optional[ProgressFn] = None,
) -> int:
    """对语料构建/增量更新向量索引。返回索引中的向量数。"""
    progress = progress or _noop
    papers = load_corpus(config.corpus_path)
    if not papers:
        progress("语料为空,跳过嵌入。请先抓取。")
        return 0

    client = SiliconFlowClient.from_env(timeout=config.timeout)
    existing: Optional[VectorIndex] = None
    if not rebuild:
        try:
            existing = VectorIndex.load(config.index_dir)
            progress(f"已有索引 {len(existing)} 向量,模型={existing.model},做增量。")
        except FileNotFoundError:
            existing = None

    progress(f"开始嵌入 {len(papers)} 篇(模型={client.embedding_model})...")
    index = VectorIndex.build(papers, client, existing=existing)
    index.save(config.index_dir)
    progress(f"索引已写入 {config.index_dir},共 {len(index)} 向量,dim={index.dim}。")
    return len(index)


def corpus_status(config: PaperAgentConfig) -> dict:
    """返回语料 + 索引的当前状态(供 web 健康检查/管理面板)。"""
    corpus_n = len(load_corpus(config.corpus_path))
    index_n = 0
    index_model = ""
    index_dim = 0
    try:
        idx = VectorIndex.load(config.index_dir)
        index_n, index_model, index_dim = len(idx), idx.model, idx.dim
    except FileNotFoundError:
        pass
    return {
        "corpus_count": corpus_n,
        "index_count": index_n,
        "index_model": index_model,
        "index_dim": index_dim,
        "data_dir": str(config.data_dir),
        "venues": list(config.venues),
        "years": f"{config.from_year}-{config.to_year}",
    }
