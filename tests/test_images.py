from __future__ import annotations

import base64
from pathlib import Path

import pytest

from gesture_agent.media.images import MAX_IMAGE_BYTES, image_path_to_data_url


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
