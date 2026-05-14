from __future__ import annotations

import re
import unicodedata


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
ASCII_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def clean_text(text: str) -> str:
    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\u3000", " ")
    return unicodedata.normalize("NFKC", text)


def normalize_query(text: str) -> str:
    return clean_text(text).strip().lower()


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def chinese_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> set[str]:
    grams: set[str] = set()
    for match in CHINESE_RE.finditer(text):
        segment = match.group(0)
        for n in range(min_n, max_n + 1):
            for idx in range(0, max(0, len(segment) - n + 1)):
                grams.add(segment[idx : idx + n])
    return grams


def tokenize(text: str) -> set[str]:
    normalized = normalize_query(text)
    tokens = {m.group(0).lower() for m in ASCII_WORD_RE.finditer(normalized)}
    tokens.update(chinese_ngrams(normalized))
    return tokens


def strip_heading_prefix(title: str) -> str:
    title = clean_text(title).strip()
    title = re.sub(r"^\d+\s*、\s*", "", title)
    title = re.sub(r"^\d+-[a-z]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\d+\s*[.-]\s*", "", title)
    title = re.sub(r"^[(（]\d+[)）]\s*", "", title)
    title = re.sub(r"^【|】$", "", title)
    title = title.strip()
    title = re.sub(r"[(（].*?[)）]", "", title).strip()
    return title
