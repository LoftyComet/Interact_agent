"""论文 RAG 子系统配置。

与 gesture_agent 的 AgentConfig 独立。字段都有合理默认值,可被
paper_agent_config.json 覆盖(可选,不存在则用默认)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_DATA_DIR = "data/papers"
DEFAULT_VENUES = ["CHI", "UIST", "DIS", "MobileHCI", "IUI"]
DEFAULT_FROM_YEAR = 2016
DEFAULT_TO_YEAR = 2025


@dataclass
class PaperAgentConfig:
    data_dir: str = DEFAULT_DATA_DIR
    venues: list[str] = field(default_factory=lambda: list(DEFAULT_VENUES))
    from_year: int = DEFAULT_FROM_YEAR
    to_year: int = DEFAULT_TO_YEAR
    require_abstract: bool = True
    top_k: int = 6
    temperature: float = 0.2
    max_tokens: int = 2000
    timeout: int = 60
    source: str = "defaults"

    @property
    def corpus_path(self) -> Path:
        return Path(self.data_dir) / "corpus.jsonl"

    @property
    def index_dir(self) -> Path:
        return Path(self.data_dir)

    @property
    def years(self) -> range:
        return range(self.from_year, self.to_year + 1)


def load_paper_config(path: Optional[str] = None) -> PaperAgentConfig:
    """加载配置。path 为空时尝试默认 paper_agent_config.json,不存在则用内置默认。"""
    config_path = Path(path) if path else Path("paper_agent_config.json")
    if not config_path.exists():
        return PaperAgentConfig()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = PaperAgentConfig(source=str(config_path))
    for key, value in data.items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)
    return cfg
