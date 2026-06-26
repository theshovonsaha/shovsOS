from __future__ import annotations

import base64
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from config.config import cfg


VALID_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
VALID_IMAGE_QUALITIES = {"auto", "low", "medium", "high"}


def _sandbox_root() -> Path:
    return Path(os.getenv("SANDBOX_DIR", "./agent_sandbox")).resolve()


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return slug[:48] or "image"


def _decode_base64_image(value: str) -> bytes:
    raw = str(value or "").strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


async def generate_image(
    *,
    prompt: str,
    model: str = "",
    size: str = "1024x1024",
    quality: str = "auto",
    background: str = "auto",
    output_format: str = "png",
) -> dict[str, Any]:
    """Generate one image and persist it under the mounted sandbox directory.

    The returned payload is intentionally UI/API friendly and does not contain
    the raw base64 image. The image is served by FastAPI's existing
    ``/sandbox`` static mount.
    """
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    if not cfg.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for image generation")

    model = str(model or cfg.IMAGE_GENERATION_MODEL or "gpt-image-1").strip()
    size = str(size or "1024x1024").strip()
    quality = str(quality or "auto").strip()
    background = str(background or "auto").strip()
    output_format = str(output_format or "png").strip().lower()
    if size not in VALID_IMAGE_SIZES:
        raise ValueError(f"unsupported image size: {size}")
    if quality not in VALID_IMAGE_QUALITIES:
        raise ValueError(f"unsupported image quality: {quality}")
    if output_format not in {"png", "jpeg", "webp"}:
        raise ValueError(f"unsupported output format: {output_format}")

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for image generation") from exc

    client = AsyncOpenAI(api_key=cfg.OPENAI_API_KEY)
    kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    # gpt-image-1 accepts quality/background/output_format. Older image
    # models may ignore or reject these, so we only pass non-default knobs.
    if quality != "auto":
        kwargs["quality"] = quality
    if background != "auto":
        kwargs["background"] = background
    if output_format != "png":
        kwargs["output_format"] = output_format

    response = await client.images.generate(**kwargs)
    data = list(getattr(response, "data", []) or [])
    if not data:
        raise RuntimeError("image provider returned no image data")
    first = data[0]
    b64_json = getattr(first, "b64_json", None)
    remote_url = getattr(first, "url", None)
    if b64_json:
        image_bytes = _decode_base64_image(str(b64_json))
    elif remote_url:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
            fetched = await http.get(str(remote_url))
            fetched.raise_for_status()
            image_bytes = fetched.content
    else:
        raise RuntimeError("image provider returned no usable image payload")

    relative_dir = Path("generated") / "images"
    target_dir = _sandbox_root() / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_slug(prompt)}-{uuid.uuid4().hex[:10]}.{output_format}"
    target = (target_dir / filename).resolve()
    if not str(target).startswith(str(_sandbox_root())):
        raise RuntimeError("unsafe image output path")
    target.write_bytes(image_bytes)
    relative_path = (relative_dir / filename).as_posix()
    return {
        "type": "image_generation_result",
        "provider": "openai",
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "path": relative_path,
        "url": f"/sandbox/{relative_path}",
        "bytes": len(image_bytes),
    }
