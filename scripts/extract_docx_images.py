"""Extract images from .docx files in data/pictures/ and build image_index.json.

Usage:
    uv run python scripts/extract_docx_images.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# Source manuscripts live directly in data/; extracted images go under pictures/.
SOURCE_DIR = DATA_DIR
OUTPUT_DIR = DATA_DIR / "pictures" / "extracted"
SUPPORTED_MIMES = {"image/png", "image/jpeg", "image/gif", "image/svg+xml"}
HEADING_STYLES = re.compile(r"^Heading|^标题|^heading", re.IGNORECASE)


def is_heading(paragraph) -> bool:
    style_name = paragraph.style.name if paragraph.style else ""
    if HEADING_STYLES.search(style_name):
        return True
    text = paragraph.text.strip()
    if re.match(r"^\d+[-\.]\s*[a-z]?\s+", text) or re.match(r"^[（(]\d+[）)]", text):
        return True
    return False


def extract_images_from_paragraph(paragraph, doc_part) -> list[tuple[bytes, str]]:
    """Return list of (image_bytes, content_type) from a paragraph."""
    images = []
    for run in paragraph.runs:
        for drawing in run.element.findall(f".//{qn('a:blip')}"):
            embed_id = drawing.get(qn("r:embed"))
            if embed_id and embed_id in doc_part.rels:
                rel = doc_part.rels[embed_id]
                blob = rel.target_part.blob
                content_type = rel.target_part.content_type
                images.append((blob, content_type))
    return images


def mime_to_ext(content_type: str) -> str:
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/svg+xml": "svg",
    }
    return mapping.get(content_type, "")


def process_docx(docx_path: Path, seen_hashes: set[str]) -> list[dict]:
    doc = Document(str(docx_path))
    doc_part = doc.part
    stem = docx_path.stem
    entries = []
    counter = 0
    current_heading = ""
    paragraphs = list(doc.paragraphs)

    for i, para in enumerate(paragraphs):
        text = para.text.strip()
        if is_heading(para) and text:
            current_heading = text
            continue

        images = extract_images_from_paragraph(para, doc_part)
        if not images:
            continue

        # In the final manuscripts the figure caption is the paragraph that
        # immediately follows the image, written as "（图…）" / "(图…)". Prefer
        # it; it describes the image far better than surrounding body text.
        caption_text = ""
        after_text = ""
        for j in range(i + 1, min(i + 4, len(paragraphs))):
            t = paragraphs[j].text.strip()
            if not t:
                continue
            if not after_text:
                after_text = t[:200]
            if re.match(r"^[（(]\s*图", t):
                caption_text = t[:300]
            break

        before_text = ""
        for j in range(i - 1, max(i - 4, -1), -1):
            t = paragraphs[j].text.strip()
            if t and not is_heading(paragraphs[j]):
                before_text = t[:200]
                break

        for blob, content_type in images:
            if content_type not in SUPPORTED_MIMES:
                continue
            content_hash = hashlib.md5(blob).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            counter += 1
            ext = mime_to_ext(content_type)
            filename = f"{stem}_{counter:03d}.{ext}"
            image_id = f"{stem}_{counter:03d}"

            annotation = caption_text or text or before_text or after_text or current_heading
            if not annotation:
                annotation = f"{stem} 图{counter}"

            (OUTPUT_DIR / filename).write_bytes(blob)
            entries.append({
                "id": image_id,
                "filename": filename,
                "source_docx": docx_path.name,
                "heading": current_heading,
                "annotation": annotation[:300],
                "terms": extract_terms(current_heading, annotation),
            })

    return entries


def extract_terms(heading: str, annotation: str) -> list[str]:
    combined = f"{heading} {annotation}"
    terms = []
    known_pattern = re.compile(
        r"(?:单击|双击|长按|按下|开关|拖拽|滑动|轻扫|甩动|翻动|捏合|旋转|"
        r"快击|缓冲|越界|连触|点击|滚动|悬停|释放|触发|反馈|"
        r"控件|按钮|滑块|开关|列表|卡片|弹窗|导航|标签|"
        r"属性|状态|位移|力度|速度|方向|角度|维度|"
        r"交互机制|控件形态|基础属性|响应逻辑|交互特性)"
    )
    for m in known_pattern.finditer(combined):
        t = m.group()
        if t not in terms:
            terms.append(t)
    heading_term = re.match(r"^\d+-[a-z]\s+(.+?)(?:[（(]|$)", heading)
    if heading_term:
        name = heading_term.group(1).strip()
        if name and name not in terms:
            terms.insert(0, name)
    return terms[:10]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    docx_files = sorted(SOURCE_DIR.glob("*.docx"))
    if not docx_files:
        print("No .docx files found in", SOURCE_DIR)
        return

    all_entries = []
    seen_hashes: set[str] = set()
    for docx_path in docx_files:
        print(f"Processing: {docx_path.name}")
        entries = process_docx(docx_path, seen_hashes)
        all_entries.extend(entries)
        print(f"  Extracted {len(entries)} images")

    index = {
        "version": 1,
        "source_files": [f.name for f in docx_files],
        "images": all_entries,
    }
    index_path = OUTPUT_DIR / "image_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal: {len(all_entries)} images → {OUTPUT_DIR}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()

