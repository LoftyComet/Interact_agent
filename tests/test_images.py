from __future__ import annotations

import base64
import re
from pathlib import Path
from urllib.parse import quote

import pytest

from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.knowledge.image_index import ImageIndex
from gesture_agent.learning.prompt_builder import format_available_images
from gesture_agent.media.images import MAX_IMAGE_BYTES, image_path_to_data_url
from web.backend.app import create_app, resolve_image_refs


def _write_png(path: Path) -> None:
    # Minimal valid 1×1 PNG
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6260000000020001e221bc330000000049454e44ae426082"
    )
    path.write_bytes(data)


def test_valid_png_returns_data_url(tmp_path: Path) -> None:
    img = tmp_path / "test.png"
    _write_png(img)
    url = image_path_to_data_url(img)
    assert url.startswith("data:image/png;base64,")
    # Decode and verify it's the original bytes
    encoded = url.split(",", 1)[1]
    assert base64.b64decode(encoded) == img.read_bytes()


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        image_path_to_data_url(tmp_path / "nonexistent.png")


def test_non_image_file_raises_value_error(tmp_path: Path) -> None:
    txt = tmp_path / "doc.txt"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported image type"):
        image_path_to_data_url(txt)


def test_oversized_file_raises_value_error(tmp_path: Path) -> None:
    img = tmp_path / "big.png"
    _write_png(img)
    with pytest.raises(ValueError, match="Image too large"):
        image_path_to_data_url(img, max_bytes=1)


def test_custom_max_bytes_allows_small_file(tmp_path: Path) -> None:
    img = tmp_path / "small.png"
    _write_png(img)
    url = image_path_to_data_url(img, max_bytes=MAX_IMAGE_BYTES)
    assert url.startswith("data:image/png;base64,")


def test_data_url_passthrough() -> None:
    encoded = base64.b64encode(b"fake-image-bytes").decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    assert image_path_to_data_url(data_url) == data_url


def test_data_url_oversized_raises_value_error() -> None:
    encoded = base64.b64encode(b"x" * 4096).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    with pytest.raises(ValueError, match="Image too large"):
        image_path_to_data_url(data_url, max_bytes=1024)


def test_data_url_without_base64_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported image data URL"):
        image_path_to_data_url("data:image/png,not-base64-data")


def test_repository_image_index_is_available_and_resolves_real_asset() -> None:
    image_index = ImageIndex.load(Path("data/pictures/extracted/image_index.json"))

    assert image_index is not None
    assert image_index.entries
    first = image_index.entries[0]
    assert (image_index.base_dir / first.filename).is_file()


def test_vector_menu_retrieval_returns_its_reference_image() -> None:
    kb = KnowledgeBase.load("data")
    query = "向量菜单是什么，怎么理解"
    terms = kb.find_terms(query)
    chunks = kb.search(query, top_k=6, prefer_terms=terms)
    image_index = ImageIndex.load(Path("data/pictures/extracted/image_index.json"))

    assert image_index is not None
    images = image_index.search(terms, chunks, top_k=3)
    assert images
    assert images[0].heading == "3-h 向量菜单"


def test_prompt_uses_url_safe_resolvable_image_reference_id() -> None:
    image_index = ImageIndex.load(Path("data/pictures/extracted/image_index.json"))
    assert image_index is not None
    image = next(item for item in image_index.entries if item.heading == "3-h 向量菜单")

    prompt = format_available_images([image])
    match = re.search(r"image:(img_[0-9a-f]{12})", prompt)

    assert match is not None
    resolved = resolve_image_refs(
        f"![向量菜单示意图](image:{match.group(1)})",
        image_index,
    )
    assert resolved.startswith("![向量菜单示意图](/api/images/")


def test_web_serves_a_repository_reference_image() -> None:
    app = create_app()
    image_index = app.config["AGENT_RUNTIME"].image_index

    assert image_index is not None
    filename = image_index.entries[0].filename
    response = app.test_client().get(f"/api/images/{quote(filename)}")
    health = app.test_client().get("/api/health").get_json()

    assert response.status_code == 200
    assert response.content_type.startswith("image/")
    assert health["image_count"] == len(image_index.entries)
