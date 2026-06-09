"""Convert the final-manuscript .docx files in data/ into the plain-text
Markdown knowledge files the KnowledgeBase parses.

The KnowledgeBase does NOT use `#` markers — it detects headings via regex on
plain-text lines (see gesture_agent/knowledge/base.py:_heading_indexes). So this
converter reproduces the exact plain-text layout the current .md files use:

  - Title style       -> first line, verbatim
  - any Heading style -> the heading text on its own line, verbatim (no `#`)
  - list paragraphs   -> prefixed with `●` (matches existing 交互特性/案例 bullets)
  - inline images     -> emitted as the figure caption only; the caption is the
                         following `（图…）` paragraph if present, else dropped
                         (images live in image_index.json, never inline in .md)
  - everything else   -> the paragraph text, verbatim

Usage:
    uv run python scripts/convert_docx_to_md.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BULLET = "●"  # ●

# new manuscript docx  ->  knowledge-base markdown filename
DOCX_TO_MD = {
    "1 序篇：基础概念.docx": "序篇_基础属性.md",
    "2 第一二篇：交互机制.docx": "交互机制.md",
    "3第三四篇：控件形态.docx": "控件形态与含义.md",
}


def is_list(paragraph) -> bool:
    return paragraph._p.find(f".//{qn('w:numPr')}") is not None


def has_image(paragraph) -> bool:
    for run in paragraph.runs:
        if run.element.findall(f".//{qn('a:blip')}"):
            return True
    return False


def is_heading(paragraph) -> bool:
    name = paragraph.style.name if paragraph.style else ""
    return name.startswith("Heading") or name == "Title"


def convert(docx_path: Path) -> str:
    doc = Document(str(docx_path))
    lines: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()

        if has_image(para):
            # The image itself is handled by the extraction pipeline. Keep only
            # an inline caption when the paragraph text itself is the caption
            # (e.g. a Heading 5 image slot carries no text -> skip).
            if text:
                lines.append(BULLET + text if is_list(para) else text)
            continue

        if not text:
            lines.append("")
            continue

        if is_heading(para):
            lines.append(text)
        elif is_list(para):
            lines.append(BULLET + text)
        else:
            lines.append(text)

    # Collapse 3+ blank lines down to at most 2, matching the current files.
    out: list[str] = []
    blanks = 0
    for ln in lines:
        if ln == "":
            blanks += 1
            if blanks <= 2:
                out.append(ln)
        else:
            blanks = 0
            out.append(ln)
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    for docx_name, md_name in DOCX_TO_MD.items():
        docx_path = DATA_DIR / docx_name
        if not docx_path.exists():
            print(f"SKIP (missing): {docx_path}")
            continue
        md = convert(docx_path)
        out_path = DATA_DIR / md_name
        out_path.write_text(md, encoding="utf-8")
        print(f"{docx_name}  ->  {md_name}  ({md.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
