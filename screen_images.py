"""Dependency-light PNG encoding and profile image crop helpers."""

import struct
import zlib


def _png_chunk(chunk_type, data):
    payload = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def _encode_frame_region_png(frame, left, top, right, bottom, max_edge):
    crop_width = right - left
    crop_height = bottom - top
    scale = min(1.0, max_edge / max(crop_width, crop_height))
    output_width = max(1, round(crop_width * scale))
    output_height = max(1, round(crop_height * scale))
    rows = bytearray()
    for output_y in range(output_height):
        source_y = top + min(crop_height - 1, int(output_y / scale))
        rows.append(0)
        for output_x in range(output_width):
            source_x = left + min(crop_width - 1, int(output_x / scale))
            blue, green, red = frame.color_at(source_x, source_y)
            rows.extend((red, green, blue))
    header = struct.pack(">IIBBBBB", output_width, output_height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
        + _png_chunk(b"IEND", b"")
    )


def first_profile_photo_png(capture, heart_point, max_edge=768):
    """Crop the first photo above its Hinge heart and encode it without disk I/O."""
    frame = capture.frame
    window = capture.window
    heart_frame_y = (heart_point[1] - window.top) * capture.scale_y
    left = round(frame.width * 0.05)
    right = round(frame.width * 0.95)
    top = round(frame.height * 0.12)
    bottom = round(min(frame.height * 0.78, heart_frame_y - frame.height * 0.045))
    if right - left < 80 or bottom - top < frame.height * 0.22:
        raise ReplyGenerationError(
            "The first profile photo was not sufficiently visible for a safe crop."
        )
    return _encode_frame_region_png(frame, left, top, right, bottom, max_edge)


def prompt_viewport_png(capture, max_edge=1400):
    """Encode the readable profile viewport for Qwen prompt rescue."""
    frame = capture.frame
    return _encode_frame_region_png(
        frame,
        round(frame.width * 0.04),
        round(frame.height * 0.07),
        round(frame.width * 0.96),
        round(frame.height * 0.91),
        max_edge,
    )
