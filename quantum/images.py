"""Bounded image checks for calibration plots. PNG and JPEG only."""

from __future__ import annotations

import base64
import binascii
import hashlib
import struct

MAX_IMAGES = 2
MAX_IMAGE_BYTES = 1_500_000
MAX_TOTAL_IMAGE_BYTES = 2_000_000
MAX_DIMENSION = 4096
MAX_PIXELS = 4_000_000

_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"


def decode_image(mime: str, encoded: str) -> tuple[bytes, str]:
    if mime not in {"image/png", "image/jpeg"}:
        raise ValueError("mime")
    if not isinstance(encoded, str) or len(encoded) > MAX_IMAGE_BYTES * 2:
        raise ValueError("size")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("encoding") from None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("size")
    if mime == "image/png":
        width, height = _png_size(raw)
    else:
        width, height = _jpeg_size(raw)
    if width < 1 or height < 1 or width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValueError("dimensions")
    if width * height > MAX_PIXELS:
        raise ValueError("dimensions")
    return raw, hashlib.sha256(raw).hexdigest()


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != _PNG or data[12:16] != b"IHDR":
        raise ValueError("mime")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:3] != _JPEG:
        raise ValueError("mime")
    index = 2
    sof = {0xC0, 0xC1, 0xC2}
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in sof:
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        if marker in {0xD8, 0xD9}:
            index += 2
            continue
        if index + 4 > len(data):
            break
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        if length < 2:
            break
        index += 2 + length
    raise ValueError("mime")
