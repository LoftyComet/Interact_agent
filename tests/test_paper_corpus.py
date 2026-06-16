"""论文语料 JSONL 读写与去重测试。"""

from paper_agent.corpus import dedupe, load_corpus, write_corpus
from paper_agent.models import Paper


def _paper(pid: str, title: str = "T", **kw) -> Paper:
    return Paper(
        id=pid,
        title=title,
        authors=kw.get("authors", ["张三", "李四"]),
        year=kw.get("year", 2023),
        venue=kw.get("venue", "UIST"),
        abstract=kw.get("abstract", "摘要内容 with unicode 标题"),
        doi=kw.get("doi", ""),
        url=kw.get("url", ""),
        citations=kw.get("citations", 0),
        concepts=kw.get("concepts", []),
    )


def test_write_load_roundtrip_preserves_unicode(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    papers = [_paper("a", "中文标题 Title"), _paper("b", "Another")]
    n = write_corpus(path, papers)
    assert n == 2

    loaded = load_corpus(path)
    assert len(loaded) == 2
    assert loaded[0].title == "中文标题 Title"
    assert loaded[0].authors == ["张三", "李四"]
    assert loaded[0].abstract == "摘要内容 with unicode 标题"


def test_load_missing_file_returns_empty(tmp_path) -> None:
    assert load_corpus(tmp_path / "nope.jsonl") == []


def test_dedupe_by_id_then_doi_then_title() -> None:
    papers = [
        _paper("x", "First"),
        _paper("x", "Dup by id"),          # 同 id,去掉
        _paper("", "Same", doi="10.1/a"),
        _paper("", "Other", doi="10.1/a"),  # 同 doi,去掉
        _paper("", "Title only"),
        _paper("", "title only"),           # 同标题(小写),去掉
    ]
    result = dedupe(papers)
    titles = [p.title for p in result]
    assert titles == ["First", "Same", "Title only"]


def test_append_mode_then_load_dedupes(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    write_corpus(path, [_paper("a")], mode="w")
    write_corpus(path, [_paper("a"), _paper("b")], mode="a")
    loaded = load_corpus(path)
    assert sorted(p.id for p in loaded) == ["a", "b"]
