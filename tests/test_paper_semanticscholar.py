"""Semantic Scholar 抓取层:字段映射与分页测试(mock HTTP,无网络)。"""

from paper_agent.sources.semanticscholar import SemanticScholarClient


SAMPLE_RAW = {
    "paperId": "abc123",
    "title": "  A Great Paper  ",
    "year": 2023,
    "venue": "ACM Symposium on User Interface Software and Technology",
    "abstract": "We propose something novel.",
    "authors": [{"name": "Alice"}, {"name": "Bob"}, {"name": ""}],
    "citationCount": 42,
    "externalIds": {"DOI": "10.1145/xyz"},
    "url": "https://www.semanticscholar.org/paper/abc123",
}


def test_to_paper_maps_fields() -> None:
    paper = SemanticScholarClient._to_paper(SAMPLE_RAW, venue_label="UIST")
    assert paper is not None
    assert paper.id == "abc123"
    assert paper.title == "A Great Paper"
    assert paper.authors == ["Alice", "Bob"]  # 空名字被过滤
    assert paper.year == 2023
    assert paper.venue == "UIST"            # 用 label,不用原始长名
    assert paper.doi == "10.1145/xyz"
    assert paper.citations == 42


def test_to_paper_skips_empty_title() -> None:
    assert SemanticScholarClient._to_paper({"title": "  "}, venue_label="UIST") is None


def test_to_paper_url_fallback_to_doi() -> None:
    raw = {"paperId": "x", "title": "T", "externalIds": {"DOI": "10.1/a"}}
    paper = SemanticScholarClient._to_paper(raw, venue_label="UIST")
    assert paper.url == "https://doi.org/10.1/a"


def test_fetch_venue_year_follows_token(monkeypatch) -> None:
    # 两页:第一页带 token,第二页无 token 终止
    pages = [
        {"data": [dict(SAMPLE_RAW, paperId="p1")], "token": "next"},
        {"data": [dict(SAMPLE_RAW, paperId="p2")], "token": None},
    ]
    calls = []

    def fake_get(self, client, path, params):
        calls.append(params.get("token"))
        return pages[len(calls) - 1]

    monkeypatch.setattr(SemanticScholarClient, "_get", fake_get)
    client = SemanticScholarClient(page_pause=0)
    papers = list(client.fetch_venue_year("UIST", "2023", label="UIST"))

    assert [p.id for p in papers] == ["p1", "p2"]
    assert calls == [None, "next"]  # 第一次无 token,第二次带 token
