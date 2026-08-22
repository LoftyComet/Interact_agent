"""Structure-aware chunking for normalized corpus documents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from .models import DocumentBlock, KnowledgeChunk, NormalizedDocument


CHUNK_SCHEMA_VERSION = "1.0"


def chunk_documents(
    documents: list[NormalizedDocument],
    *,
    max_chars: int = 1000,
    overlap_chars: int = 120,
) -> tuple[list[KnowledgeChunk], dict]:
    """Chunk documents at heading seams, splitting only oversized sections."""

    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(_chunk_document(document, max_chars=max_chars, overlap_chars=overlap_chars))
    chunks = _link_neighbors(chunks)
    report = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "average_chars": round(sum(chunk.char_count for chunk in chunks) / len(chunks), 1) if chunks else 0,
        "largest_chunk_chars": max((chunk.char_count for chunk in chunks), default=0),
        "chunks_over_limit": sum(chunk.char_count > max_chars for chunk in chunks),
        "by_role": _count_by_role(chunks),
    }
    return chunks, report


def load_documents(path: str | Path) -> list[NormalizedDocument]:
    documents: list[NormalizedDocument] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        blocks = [DocumentBlock(**block) for block in payload.pop("blocks", [])]
        documents.append(NormalizedDocument(blocks=blocks, **payload))
    return documents


def write_chunks(chunks: list[KnowledgeChunk], path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return output


def _chunk_document(document: NormalizedDocument, *, max_chars: int, overlap_chars: int) -> list[KnowledgeChunk]:
    heading_stack: list[str] = []
    section_blocks: list[DocumentBlock] = []
    section_heading_path: list[str] = []
    results: list[KnowledgeChunk] = []

    def flush_section() -> None:
        nonlocal section_blocks
        if not section_blocks:
            return
        results.extend(
            _chunk_section(
                document,
                section_heading_path,
                section_blocks,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
        section_blocks = []

    for block in document.blocks:
        if block.kind == "heading":
            flush_section()
            level = block.heading_level or min(len(heading_stack) + 1, 6)
            heading_stack = heading_stack[: max(level - 1, 0)]
            heading_stack.append(block.text)
            section_heading_path = list(heading_stack)
            section_blocks = [block]
        else:
            if not section_blocks:
                section_heading_path = list(heading_stack)
            section_blocks.append(block)
    flush_section()
    return results


def _chunk_section(
    document: NormalizedDocument,
    heading_path: list[str],
    blocks: list[DocumentBlock],
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[KnowledgeChunk]:
    units: list[tuple[str, dict]] = []
    for block in blocks:
        if block.kind == "heading":
            continue
        for piece in _split_oversized(block.text, max_chars):
            units.append((piece, block.source_locator))
    if not units:
        return []

    groups: list[tuple[str, list[dict]]] = []
    current_parts: list[str] = []
    current_locators: list[dict] = []
    for text, locator in units:
        candidate = "\n\n".join([*current_parts, text])
        if current_parts and len(candidate) > max_chars:
            complete = "\n\n".join(current_parts)
            groups.append((complete, _dedupe_locators(current_locators)))
            available_overlap = max(max_chars - len(text) - 2, 0)
            overlap = _tail_at_sentence(complete, min(overlap_chars, available_overlap))
            current_parts = [overlap, text] if overlap else [text]
            current_locators = [current_locators[-1], locator] if overlap and current_locators else [locator]
        else:
            current_parts.append(text)
            current_locators.append(locator)
    if current_parts:
        groups.append(("\n\n".join(current_parts), _dedupe_locators(current_locators)))

    section_title = heading_path[-1] if heading_path else document.title
    chunks: list[KnowledgeChunk] = []
    for part_index, (text, locators) in enumerate(groups):
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = "\0".join([document.id, *heading_path, str(part_index), content_hash])
        chunk_id = "chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        path_prefix = " > ".join([document.title, *heading_path])
        retrieval_text = f"{path_prefix}\n{text}" if path_prefix else text
        chunks.append(
            KnowledgeChunk(
                schema_version=CHUNK_SCHEMA_VERSION,
                id=chunk_id,
                document_id=document.id,
                file_id=document.file_id,
                title=section_title,
                heading_path=list(heading_path),
                text=text,
                retrieval_text=retrieval_text,
                source_path=document.source_path,
                source_locators=locators,
                role=document.role,
                authority=document.authority,
                content_sha256=content_hash,
                char_count=len(text),
            )
        )
    return chunks


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [item.strip() for item in re.split(r"(?<=[。！？；.!?;])", text) if item.strip()]
    if len(sentences) <= 1:
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
    atomic: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            atomic.append(sentence)
        else:
            atomic.extend(sentence[index : index + max_chars] for index in range(0, len(sentence), max_chars))
    pieces: list[str] = []
    current = ""
    for sentence in atomic:
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces


def _tail_at_sentence(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    tail = text[-limit:]
    for marker in ("。", "！", "？", "；", ".", "!", "?", ";", "\n"):
        position = tail.find(marker)
        if 0 <= position < len(tail) - 1:
            return tail[position + 1 :].strip()
    return tail.strip()


def _dedupe_locators(locators: list[dict]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for locator in locators:
        key = json.dumps(locator, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(locator)
    return result


def _link_neighbors(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    by_document: dict[str, list[KnowledgeChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)
    linked: list[KnowledgeChunk] = []
    for document_chunks in by_document.values():
        for index, chunk in enumerate(document_chunks):
            linked.append(
                replace(
                    chunk,
                    previous_chunk_id=document_chunks[index - 1].id if index > 0 else "",
                    next_chunk_id=document_chunks[index + 1].id if index + 1 < len(document_chunks) else "",
                )
            )
    order = {chunk.id: index for index, chunk in enumerate(chunks)}
    return sorted(linked, key=lambda chunk: order[chunk.id])


def _count_by_role(chunks: list[KnowledgeChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.role] = counts.get(chunk.role, 0) + 1
    return dict(sorted(counts.items()))
