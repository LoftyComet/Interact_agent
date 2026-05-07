from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Union


def image_path_to_data_url(path: Union[str, Path]) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
