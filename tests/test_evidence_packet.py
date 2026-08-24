from copy import deepcopy

from gesture_agent.core.models import SourceChunk
from gesture_agent.learning.evidence_packet import build_evidence_packet


def test_evidence_packet_is_verbatim_traceable_and_does_not_mutate_chunks() -> None:
    chunks = [SourceChunk(
        id="chunk-1",
        title="力属性",
        source="book.docx",
        start_line=12,
        end_line=14,
        text="力属性包括大小和方向。后一句仍是原文。",
        layer="basic_property",
        terms=["力属性"],
        score=0.8,
    )]
    original = deepcopy(chunks)

    packet = build_evidence_packet(chunks, max_chars=12)

    assert chunks == original
    assert packet.records[0].excerpt in chunks[0].text
    assert packet.records[0].chunk_id == "chunk-1"
    assert "索引块：chunk-1" in packet.render()
    assert "原文指纹" in packet.render()
