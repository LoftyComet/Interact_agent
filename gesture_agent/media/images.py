from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Union

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def _validate_data_url(data_url: str, max_bytes: int) -> str:
    """Pass through a browser-supplied image data URL after sanity-checking it.

    The web frontend reads picked files with FileReader.readAsDataURL and sends
    the resulting ``data:image/...;base64,...`` string in the same ``images``
    array that used to carry server-local paths. Validate the decoded size so a
    data URL can't bypass the disk path's size limit.
    """
    header, _, encoded = data_url.partition(",")
    if "base64" not in header:
        raise ValueError("Unsupported image data URL: expected base64 encoding")
    # base64 length * 3/4 approximates decoded bytes; subtract padding.
    padding = encoded.count("=")
    decoded_size = (len(encoded) * 3) // 4 - padding
    if decoded_size > max_bytes:
        raise ValueError(
            f"Image too large: {decoded_size / 1024 / 1024:.1f} MB "
            f"(limit {max_bytes / 1024 / 1024:.0f} MB)"
        )
    return data_url


def image_path_to_data_url(path: Union[str, Path], max_bytes: int = MAX_IMAGE_BYTES) -> str:
    # Browser uploads arrive already encoded as data URLs — return them as-is.
    if isinstance(path, str) and path.startswith("data:image/"):
        return _validate_data_url(path, max_bytes)
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {image_path}")
    file_size = image_path.stat().st_size
    if file_size > max_bytes:
        raise ValueError(
            f"Image too large: {file_size / 1024 / 1024:.1f} MB "
            f"(limit {max_bytes / 1024 / 1024:.0f} MB): {image_path}"
        )
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
