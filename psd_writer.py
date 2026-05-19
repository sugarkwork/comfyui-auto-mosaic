"""Minimal PSD writer.

Ported from the MyPsdWriter C# implementation at /mnt/f/ai/mypsd. Produces
Photoshop-compatible PSD files with multiple RGBA layers and an RGB composite
preview. Uses only the Python stdlib plus numpy/PIL (already required by the
rest of the package), so no extra runtime dependency is needed.

References: Adobe Photoshop File Formats Specification
(https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/).
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


COMPRESSION_RAW = 0
COMPRESSION_RLE = 1
COMPRESSION_ZIP = 2
COMPRESSION_ZIP_PREDICT = 3

_COMPRESSION_NAMES = {
    "raw": COMPRESSION_RAW,
    "rle": COMPRESSION_RLE,
    "zip": COMPRESSION_ZIP,
    "zip_predict": COMPRESSION_ZIP_PREDICT,
}


class PsdLayer:
    """Single PSD layer at full canvas coordinates.

    `pixels_rgba` is a flat bytes/bytearray of length width*height*4 in row-major
    RGBA order, matching the C# `PixelsRgba` field.
    """

    __slots__ = ("name", "left", "top", "width", "height", "pixels_rgba", "opacity", "visible")

    def __init__(self, name: str, left: int, top: int, width: int, height: int,
                 pixels_rgba: bytes, opacity: int = 255, visible: bool = True):
        self.name = name
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)
        self.pixels_rgba = pixels_rgba
        self.opacity = int(opacity) & 0xFF
        self.visible = bool(visible)

    @classmethod
    def from_pil(cls, name: str, image: Image.Image, left: int = 0, top: int = 0,
                 opacity: int = 255, visible: bool = True) -> "PsdLayer":
        rgba = image.convert("RGBA")
        return cls(
            name=name, left=left, top=top,
            width=rgba.width, height=rgba.height,
            pixels_rgba=rgba.tobytes(),
            opacity=opacity, visible=visible,
        )

    def validate(self, canvas_width: int, canvas_height: int) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Layer name is required.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Layer width and height must be > 0.")
        if (self.left < 0 or self.top < 0 or
                self.left + self.width > canvas_width or
                self.top + self.height > canvas_height):
            raise ValueError("Layer rectangle must stay inside the canvas.")
        expected = self.width * self.height * 4
        if len(self.pixels_rgba) != expected:
            raise ValueError(f"pixels_rgba length must be {expected}, got {len(self.pixels_rgba)}.")


def _resolve_compression(compression) -> int:
    if isinstance(compression, int):
        return compression
    key = str(compression).lower()
    if key not in _COMPRESSION_NAMES:
        raise ValueError(f"Unknown compression: {compression}")
    return _COMPRESSION_NAMES[key]


def _u16_be(value: int) -> bytes:
    return struct.pack(">H", value & 0xFFFF)


def _i16_be(value: int) -> bytes:
    return struct.pack(">h", value)


def _i32_be(value: int) -> bytes:
    return struct.pack(">i", value)


def _u32_be(value: int) -> bytes:
    return struct.pack(">I", value & 0xFFFFFFFF)


def _rle_encode_line(line: bytes) -> bytes:
    """PackBits-style RLE encoding for a single scanline.

    Mirrors PsdCompression.RleEncodeLine in the C# reference.
    """
    out = bytearray()
    n = len(line)
    i = 0
    while i < n:
        run_length = 1
        while (i + run_length < n
               and line[i] == line[i + run_length]
               and run_length < 128):
            run_length += 1

        if run_length >= 3:
            out.append((257 - run_length) & 0xFF)
            out.append(line[i])
            i += run_length
            continue

        literal_length = 0
        while i + literal_length < n and literal_length < 128:
            ahead = i + literal_length
            has_dup = (ahead + 2 < n
                       and line[ahead] == line[ahead + 1]
                       and line[ahead] == line[ahead + 2])
            if has_dup:
                break
            literal_length += 1

        out.append((literal_length - 1) & 0xFF)
        out.extend(line[i:i + literal_length])
        i += literal_length

    return bytes(out)


def _compress_channel_raw(raw: bytes) -> bytes:
    return _u16_be(COMPRESSION_RAW) + raw


def _compress_channel_rle(raw: bytes, width: int, height: int) -> bytes:
    body = bytearray()
    body.extend(_u16_be(COMPRESSION_RLE))

    counts = []
    lines = []
    for y in range(height):
        line = raw[y * width:(y + 1) * width]
        encoded = _rle_encode_line(line)
        counts.append(len(encoded))
        lines.append(encoded)

    for c in counts:
        body.extend(_u16_be(c))
    for ln in lines:
        body.extend(ln)
    return bytes(body)


def _compress_channel_zip(raw: bytes, predict: bool, width: int, height: int) -> bytes:
    data = raw
    if predict:
        data = _prediction_encode(raw, width, height)
    payload = zlib.compress(data, 9)
    method = COMPRESSION_ZIP_PREDICT if predict else COMPRESSION_ZIP
    return _u16_be(method) + payload


def _prediction_encode(raw: bytes, width: int, height: int) -> bytes:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width).copy()
    if width > 1:
        arr[:, 1:] = (arr[:, 1:].astype(np.int16) - arr[:, :-1].astype(np.int16)).astype(np.uint8)
    return arr.tobytes()


def _compress_channel(raw: bytes, width: int, height: int, method: int) -> bytes:
    if method == COMPRESSION_RAW:
        return _compress_channel_raw(raw)
    if method == COMPRESSION_RLE:
        return _compress_channel_rle(raw, width, height)
    if method == COMPRESSION_ZIP:
        return _compress_channel_zip(raw, predict=False, width=width, height=height)
    if method == COMPRESSION_ZIP_PREDICT:
        return _compress_channel_zip(raw, predict=True, width=width, height=height)
    raise ValueError(f"Unsupported compression method: {method}")


def _compress_global_image_data(channels: Sequence[bytes], width: int, height: int, method: int) -> bytes:
    """Compress the merged image data block (a single 2-byte method header, then all channels)."""
    body = bytearray()
    body.extend(_u16_be(method))

    if method == COMPRESSION_RAW:
        for ch in channels:
            body.extend(ch)
        return bytes(body)

    if method == COMPRESSION_RLE:
        counts = []
        lines = []
        for ch in channels:
            for y in range(height):
                line = ch[y * width:(y + 1) * width]
                encoded = _rle_encode_line(line)
                counts.append(len(encoded))
                lines.append(encoded)
        for c in counts:
            body.extend(_u16_be(c))
        for ln in lines:
            body.extend(ln)
        return bytes(body)

    combined = bytearray()
    for ch in channels:
        if method == COMPRESSION_ZIP_PREDICT:
            combined.extend(_prediction_encode(ch, width, height))
        else:
            combined.extend(ch)
    body.extend(zlib.compress(bytes(combined), 9))
    return bytes(body)


def _extract_channel(pixels_rgba: bytes, component: int) -> bytes:
    arr = np.frombuffer(pixels_rgba, dtype=np.uint8).reshape(-1, 4)
    return arr[:, component].tobytes()


def _write_pascal_string(value: str, pad_multiple: int) -> bytes:
    encoded = value.encode("ascii", errors="replace")
    if len(encoded) > 255:
        encoded = encoded[:255]
    out = bytearray()
    out.append(len(encoded))
    out.extend(encoded)
    total = 1 + len(encoded)
    padded = ((total + pad_multiple - 1) // pad_multiple) * pad_multiple
    out.extend(b"\x00" * (padded - total))
    return bytes(out)


def _write_layer_record(layer: PsdLayer, channel_blocks: List[bytes], method: int) -> bytes:
    out = bytearray()
    out.extend(_i32_be(layer.top))
    out.extend(_i32_be(layer.left))
    out.extend(_i32_be(layer.top + layer.height))
    out.extend(_i32_be(layer.left + layer.width))

    out.extend(_i16_be(4))  # A, R, G, B

    alpha = _extract_channel(layer.pixels_rgba, 3)
    red = _extract_channel(layer.pixels_rgba, 0)
    green = _extract_channel(layer.pixels_rgba, 1)
    blue = _extract_channel(layer.pixels_rgba, 2)

    for ch_id, raw in ((-1, alpha), (0, red), (1, green), (2, blue)):
        block = _compress_channel(raw, layer.width, layer.height, method)
        out.extend(_i16_be(ch_id))
        out.extend(_u32_be(len(block)))
        channel_blocks.append(block)

    out.extend(b"8BIM")
    out.extend(b"norm")
    out.append(layer.opacity)
    out.append(0)  # clipping

    flags = 0
    if not layer.visible:
        flags |= 0b0000_0010
    out.append(flags)
    out.append(0)  # filler

    extra = bytearray()
    extra.extend(_u32_be(0))  # layer mask data length
    extra.extend(_u32_be(0))  # layer blending ranges length
    extra.extend(_write_pascal_string(layer.name, 4))

    out.extend(_u32_be(len(extra)))
    out.extend(extra)

    return bytes(out)


def _write_layer_info(layers: Sequence[PsdLayer], method: int) -> bytes:
    section = bytearray()
    section.extend(_i16_be(len(layers)))

    channel_blocks: List[bytes] = []
    for layer in layers:
        section.extend(_write_layer_record(layer, channel_blocks, method))

    for block in channel_blocks:
        section.extend(block)

    if len(section) & 1:
        section.append(0)

    out = bytearray()
    out.extend(_u32_be(len(section)))
    out.extend(section)
    return bytes(out)


def _alpha_composite_layers(width: int, height: int, layers: Iterable[PsdLayer]) -> Image.Image:
    """Build the canvas-sized RGB composite preview by alpha-compositing layers bottom-up."""
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for layer in layers:
        if not layer.visible:
            continue
        layer_img = Image.frombytes("RGBA", (layer.width, layer.height), layer.pixels_rgba)
        if layer.opacity != 255:
            # Scale alpha by opacity (premultiplied feel: just multiply A).
            r, g, b, a = layer_img.split()
            scaled = a.point(lambda v, op=layer.opacity: (v * op) // 255)
            layer_img = Image.merge("RGBA", (r, g, b, scaled))
        if layer.left == 0 and layer.top == 0 and layer.width == width and layer.height == height:
            canvas = Image.alpha_composite(canvas, layer_img)
        else:
            padded = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            padded.paste(layer_img, (layer.left, layer.top))
            canvas = Image.alpha_composite(canvas, padded)
    return canvas.convert("RGB")


def write_psd(layers: Sequence[PsdLayer], output_path: str, width: int, height: int,
              compression: str = "rle", composite_image: Optional[Image.Image] = None) -> None:
    """Write a PSD file to disk.

    Args:
        layers: layers in bottom-to-top order.
        output_path: destination path.
        width, height: canvas size.
        compression: "raw", "rle", "zip", or "zip_predict". Defaults to "rle"
            for best third-party compatibility (Photoshop/Affinity/CSP).
        composite_image: optional pre-computed RGB preview. If omitted, computed
            via PIL alpha_composite from the layers.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Canvas size must be > 0.")
    for layer in layers:
        layer.validate(width, height)

    method = _resolve_compression(compression)

    if composite_image is None:
        composite_image = _alpha_composite_layers(width, height, layers)
    else:
        composite_image = composite_image.convert("RGB")
        if composite_image.size != (width, height):
            composite_image = composite_image.resize((width, height))

    buf = io.BytesIO()

    # File header: RGB / 3ch / 8bpc (matches C# header writing 3 channels even though layers carry RGBA).
    buf.write(b"8BPS")
    buf.write(_u16_be(1))
    buf.write(b"\x00" * 6)
    buf.write(_u16_be(3))         # channel count for global image
    buf.write(_u32_be(height))
    buf.write(_u32_be(width))
    buf.write(_u16_be(8))         # depth
    buf.write(_u16_be(3))         # color mode = RGB

    # Color mode data: empty
    buf.write(_u32_be(0))

    # Image resources: empty
    buf.write(_u32_be(0))

    # Layer & mask information section
    layer_info_block = _write_layer_info(layers, method)
    global_mask_info = _u32_be(0)
    section = layer_info_block + global_mask_info
    buf.write(_u32_be(len(section)))
    buf.write(section)

    # Composite image data (3 channels: R, G, B)
    composite_arr = np.array(composite_image, dtype=np.uint8)  # (H, W, 3)
    channels = [
        composite_arr[:, :, 0].tobytes(),
        composite_arr[:, :, 1].tobytes(),
        composite_arr[:, :, 2].tobytes(),
    ]
    buf.write(_compress_global_image_data(channels, width, height, method))

    with open(output_path, "wb") as fh:
        fh.write(buf.getvalue())


def write_psd_from_pil_layers(pil_layers: Sequence[dict], output_path: str,
                              compression: str = "rle",
                              composite_image: Optional[Image.Image] = None) -> None:
    """Convenience wrapper for the {'name': str, 'image': PIL.Image} dict shape used elsewhere.

    All layers must share the same size; that size becomes the canvas.
    """
    if not pil_layers:
        raise ValueError("pil_layers is empty.")

    first = pil_layers[0]["image"]
    width, height = first.size

    psd_layers: List[PsdLayer] = []
    for info in pil_layers:
        img = info["image"]
        if img.size != (width, height):
            raise ValueError("All layers must share the same size as the canvas.")
        psd_layers.append(PsdLayer.from_pil(name=info.get("name", "Layer"), image=img))

    write_psd(psd_layers, output_path, width, height,
              compression=compression, composite_image=composite_image)
