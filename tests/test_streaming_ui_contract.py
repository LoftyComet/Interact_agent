from __future__ import annotations

from pathlib import Path

from gesture_agent.knowledge.image_index import ImageIndex
from web.backend.app import stream_answer_payload


FRONTEND_JS = Path("web/frontend/app.js")


def test_stream_answer_payload_resolves_images_and_includes_blocks() -> None:
    image_index = ImageIndex.load(Path("data/pictures/extracted/image_index.json"))
    assert image_index is not None
    image = next(item for item in image_index.entries if item.heading == "3-h 向量菜单")
    answer = (
        "<!-- ixdl-answer-block:corpus_evidence -->\n\n"
        f"![向量菜单](image:{image.reference_id})\n\n定义。[1]"
    )

    payload = stream_answer_payload(answer, image_index)

    assert payload["text"].startswith("<!-- ixdl-answer-block:corpus_evidence -->")
    assert "/api/images/" in payload["text"]
    assert payload["answer_blocks"][0]["type"] == "corpus_evidence"
    assert "/api/images/" in payload["answer_blocks"][0]["markdown"]


def test_dictionary_stream_handles_replace_and_verification_events() -> None:
    source = FRONTEND_JS.read_text(encoding="utf-8")
    dictionary_stream = source[
        source.index("async function sendStream") : source.index(
            "els.imageFiles.addEventListener"
        )
    ]
    replacement_renderer = dictionary_stream[
        dictionary_stream.index("function renderAnswerReplacement") : dictionary_stream.index(
            "function handleEvent"
        )
    ]

    assert 'case "replace":' in dictionary_stream
    assert "renderAnswerReplacement(payload);" in dictionary_stream
    assert 'case "verification":' in dictionary_stream
    assert 'answerText = payload.text || "";' in replacement_renderer
    assert "setBubbleAnswerBlocks(bubble, payload.answer_blocks)" in replacement_renderer
    assert "scheduleAnswerRender()" in dictionary_stream
