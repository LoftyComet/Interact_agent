"""Semantic Scholar Graph API 抓取层。

用 bulk search 端点按会议 + 年份抓取论文的元数据 + 摘要。OpenAlex 对 ACM
会议的 venue 归一化太差(实测 CHI 全部 source 仅 ~1700 篇且按年份割裂),故
选用 Semantic Scholar 作为主源:venue 过滤精确、带摘要/作者/引用数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from gesture_agent.settings.env import get_env
from ..models import Paper

# 各会议在 Semantic Scholar 中的 venue 检索名(实测命中)。
# 值为列表:同一会议可能对应多个 venue 字符串,逐个抓取后合并。
VENUES: dict[str, list[str]] = {
    "CHI": ["CHI"],
    "UIST": ["UIST"],
    "DIS": ["DIS"],
    "MobileHCI": ["MobileHCI"],
    "IUI": ["IUI"],
    # CSCW 近年论文多归在 PACMHCI 期刊下,venue 边界复杂,作为可选项。
    "CSCW": ["Proc. ACM Hum. Comput. Interact."],
}

BULK_FIELDS = "title,year,venue,abstract,authors,citationCount,externalIds,url"
DEFAULT_BASE_URL = "https://api.semanticscholar.org/graph/v1"


@dataclass
class SemanticScholarClient:
    api_key: Optional[str] = None
    timeout: int = 40
    base_url: str = DEFAULT_BASE_URL
    max_retries: int = 5
    page_pause: float = 1.0

    @classmethod
    def from_env(cls, timeout: Optional[int] = None) -> "SemanticScholarClient":
        return cls(
            api_key=get_env("S2_API_KEY"),
            timeout=timeout or int(get_env("S2_TIMEOUT", "40") or "40"),
        )

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def fetch(self, venues: list[str], years: range) -> Iterator[Paper]:
        """抓取多个会议在指定年份范围内的全部论文。

        venues 为 VENUES 的键(如 "CHI");未知键按字面量当作 venue 名。
        """
        year_spec = f"{years.start}-{years.stop - 1}" if len(years) > 1 else str(years.start)
        for venue_key in venues:
            for venue_name in VENUES.get(venue_key, [venue_key]):
                yield from self.fetch_venue_year(venue_name, year_spec, label=venue_key)

    def fetch_venue_year(
        self, venue: str, year_spec: str, *, label: Optional[str] = None
    ) -> Iterator[Paper]:
        """抓取单个 venue 在 year_spec(如 '2016-2025' 或 '2023')的论文。"""
        params = {
            "venue": venue,
            "year": year_spec,
            "fields": BULK_FIELDS,
        }
        token: Optional[str] = None
        with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
            while True:
                query = dict(params)
                if token:
                    query["token"] = token
                payload = self._get(client, "/paper/search/bulk", query)
                for raw in payload.get("data") or []:
                    paper = self._to_paper(raw, venue_label=label or venue)
                    if paper is not None:
                        yield paper
                token = payload.get("token")
                if not token:
                    break
                time.sleep(self.page_pause)

    def _get(self, client: httpx.Client, path: str, params: dict) -> dict:
        """带指数退避的 GET。遇 429/5xx 重试。"""
        delay = 2.0
        last_exc: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                resp = client.get(self.base_url + path, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Semantic Scholar 请求多次失败:{path}")

    @staticmethod
    def _to_paper(raw: dict, *, venue_label: str) -> Optional[Paper]:
        """把 Semantic Scholar 的一条记录映射为 Paper。无标题则跳过。"""
        title = (raw.get("title") or "").strip()
        if not title:
            return None
        authors = [a.get("name", "").strip() for a in (raw.get("authors") or []) if a.get("name")]
        external = raw.get("externalIds") or {}
        doi = external.get("DOI", "") or ""
        return Paper(
            id=raw.get("paperId", "") or doi or title.lower(),
            title=title,
            authors=authors,
            year=raw.get("year"),
            venue=venue_label,
            abstract=(raw.get("abstract") or "").strip(),
            doi=doi,
            url=raw.get("url", "") or (f"https://doi.org/{doi}" if doi else ""),
            citations=int(raw.get("citationCount", 0) or 0),
            concepts=[],
        )
