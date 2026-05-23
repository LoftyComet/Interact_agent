from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Union

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def image_path_to_data_url(path: Union[str, Path], max_bytes: int = MAX_IMAGE_BYTES) -> str:
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
