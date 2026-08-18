import math
import json
import os
import re
import struct
import sys
import threading
import time
import tkinter as tk
import zlib
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button, Controller

from profile_reply import (
    ProfileScanError,
    ProfileScanner,
    find_text_target,
    prompt_is_visible,
    recognize_text,
    recover_vision_prompt_target,
    reply_is_visible,
    reply_is_visible_near,
    viewport_similarity,
)
from reply_generation import (
    OllamaReplyGenerator,
    ReplyGenerationError,
    ReplyGenerator,
    TONE_INSTRUCTIONS,
    random_pickup_line,
    validate_fallback_pickup_line,
)

try:
    import AppKit
    import objc
    import Quartz
    import Vision
except ImportError:
    AppKit = None
    objc = None
    Quartz = None
    Vision = None

try:
    import ScreenCaptureKit
except ImportError:
    ScreenCaptureKit = None


IPHONE_MIRRORING_BUNDLE_ID = "com.apple.ScreenContinuity"
SCREEN_RECORDING_SETTINGS_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
)
PROMPT_FAILURE_LOG = (
    Path.home() / "Library" / "Logs" / "The Scoop UP" / "prompt_reply_failures.jsonl"
)


def format_elapsed_time(seconds):
    """Format a monotonic duration for compact status messages."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def native_autorelease_pool():
    """Drain temporary PyObjC objects at the end of one bounded operation."""
    return objc.autorelease_pool() if objc is not None else nullcontext()

# Normalized from the supplied Hinge control: a white heart cutout inside a
# solid black circular button. Keeping this as a compact binary signature makes
# detection independent of screen scale and avoids a fragile external asset.
HINGE_HEART_BUTTON_TEMPLATE = (
    "........########........",
    "......############......",
    ".....##############.....",
    "....################....",
    "...##################...",
    "..####################..",
    ".######################.",
    ".#######........#######.",
    ".######..........######.",
    "#######.########.#######",
    "#######.########..######",
    "#######.########.#######",
    "#######..######..#######",
    "#######...####...#######",
    "########...#############",
    "#########...############",
    ".#########...##########.",
    ".##########.###########.",
    "..####################..",
    "..####################..",
    "...##################...",
    "....################....",
    ".....##############.....",
    ".......##########.......",
)


class ScreenRecordingPermissionError(RuntimeError):
    """Raised when macOS blocks capture of the mirroring window."""


class PixelFrame:
    """A dependency-light RGB view over a Quartz screen capture."""

    def __init__(self, width, height, bytes_per_row, pixels):
        self.width = int(width)
        self.height = int(height)
        self.bytes_per_row = int(bytes_per_row)
        self.pixels = pixels

    def color_at(self, x, y):
        x = max(0, min(self.width - 1, int(x)))
        y = max(0, min(self.height - 1, int(y)))
        offset = y * self.bytes_per_row + x * 4
        # Quartz commonly returns BGRA. Only relative channel comparisons are
        # used, and red/blue are interchangeable for the green Tinder target.
        return tuple(self.pixels[offset:offset + 3])


@dataclass(frozen=True)
class MirroringWindow:
    window_id: int
    owner_pid: int
    owner_name: str
    window_name: str
    left: float
    top: float
    width: float
    height: float


@dataclass
class WindowCapture:
    window: MirroringWindow
    image: object
    frame: PixelFrame

    def ensure_vision_image(self):
        """Build a process-local CGImage so OCR never retains WindowServer transport."""
        if self.image is not None:
            return self.image
        if Quartz is None or AppKit is None:
            raise RuntimeError("Native macOS image support is unavailable.")
        data = AppKit.NSData.dataWithBytes_length_(
            self.frame.pixels,
            len(self.frame.pixels),
        )
        provider = Quartz.CGDataProviderCreateWithCFData(data)
        color_space = Quartz.CGColorSpaceCreateDeviceRGB()
        bitmap_info = (
            Quartz.kCGImageAlphaPremultipliedFirst
            | Quartz.kCGBitmapByteOrder32Little
        )
        self.image = Quartz.CGImageCreate(
            self.frame.width,
            self.frame.height,
            8,
            32,
            self.frame.bytes_per_row,
            color_space,
            bitmap_info,
            provider,
            None,
            False,
            Quartz.kCGRenderingIntentDefault,
        )
        return self.image

    @property
    def scale_x(self):
        return self.frame.width / self.window.width

    @property
    def scale_y(self):
        return self.frame.height / self.window.height

    def desktop_point(self, frame_x, frame_y):
        return (
            round(self.window.left + frame_x / self.scale_x),
            round(self.window.top + frame_y / self.scale_y),
        )


def _window_bundle_id(owner_pid):
    if AppKit is None or not owner_pid:
        return ""
    app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(owner_pid)
    return str(app.bundleIdentifier() or "") if app else ""


def _window_score(owner_name, window_name, bundle_id, width, height):
    """Score window metadata without accepting similarly named development tools."""
    text = f"{owner_name} {window_name}".casefold()
    if "simulator" in text or "xcode" in text:
        return -1

    score = 0
    if bundle_id == IPHONE_MIRRORING_BUNDLE_ID:
        score += 200
    if "iphone mirroring" in text:
        score += 120
    elif "iphone" in text and "mirror" in text:
        score += 90
    if height > width * 1.25:
        score += 15
    if width >= 200 and height >= 300:
        score += 10
    return score


def find_iphone_mirroring_window():
    """Locate the visible iPhone Mirroring content window from macOS metadata."""
    if Quartz is None:
        raise RuntimeError("macOS Quartz support is not installed.")

    options = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements
    )
    windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    candidates = []

    for info in windows:
        layer = int(info.get(Quartz.kCGWindowLayer, 0))
        alpha = float(info.get(Quartz.kCGWindowAlpha, 1.0))
        if layer != 0 or alpha <= 0:
            continue

        bounds = info.get(Quartz.kCGWindowBounds, {})
        left = float(bounds.get("X", 0))
        top = float(bounds.get("Y", 0))
        width = float(bounds.get("Width", 0))
        height = float(bounds.get("Height", 0))
        if width < 150 or height < 250:
            continue

        owner_pid = int(info.get(Quartz.kCGWindowOwnerPID, 0))
        owner_name = str(info.get(Quartz.kCGWindowOwnerName, ""))
        window_name = str(info.get(Quartz.kCGWindowName, ""))
        bundle_id = _window_bundle_id(owner_pid)
        score = _window_score(owner_name, window_name, bundle_id, width, height)
        if score < 80:
            continue

        window = MirroringWindow(
            window_id=int(info.get(Quartz.kCGWindowNumber, 0)),
            owner_pid=owner_pid,
            owner_name=owner_name,
            window_name=window_name,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        candidates.append((score, width * height, window))

    if not candidates:
        raise RuntimeError(
            "iPhone Mirroring window not found. Open it, connect the iPhone, and keep it visible."
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


class ScreenCaptureBackend:
    """Reuse a ScreenCaptureKit window filter and immediately copy each frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._window_id = None
        self._window_size = None
        self._filter = None
        self._configuration = None

    @staticmethod
    def _await(starter, timeout=10):
        completed = threading.Event()
        result = {}

        def callback(value, error):
            result["value"] = value
            result["error"] = error
            completed.set()

        starter(callback)
        if not completed.wait(timeout):
            raise RuntimeError("ScreenCaptureKit timed out while capturing iPhone Mirroring.")
        if result.get("error") is not None:
            raise ScreenRecordingPermissionError(
                f"ScreenCaptureKit could not capture iPhone Mirroring: {result['error']}"
            )
        return result.get("value")

    def _configure(self, window):
        window_size = (round(window.width), round(window.height))
        if (
            self._window_id == window.window_id
            and self._window_size == window_size
            and self._filter is not None
        ):
            return
        if ScreenCaptureKit is None:
            raise RuntimeError(
                "ScreenCaptureKit support is not installed. Reinstall requirements.txt."
            )
        shareable_content = ScreenCaptureKit.SCShareableContent
        content = self._await(
            lambda callback: shareable_content.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                True,
                True,
                callback,
            )
        )
        shared_window = next(
            (
                candidate
                for candidate in content.windows()
                if int(candidate.windowID()) == window.window_id
            ),
            None,
        )
        if shared_window is None:
            raise RuntimeError("iPhone Mirroring is not available to ScreenCaptureKit.")
        capture_filter = ScreenCaptureKit.SCContentFilter.alloc().initWithDesktopIndependentWindow_(
            shared_window
        )
        scale = float(capture_filter.pointPixelScale() or 1.0)
        configuration = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
        configuration.setWidth_(round(window.width * scale))
        configuration.setHeight_(round(window.height * scale))
        configuration.setShowsCursor_(False)
        configuration.setIgnoreShadowsSingleWindow_(True)
        self._window_id = window.window_id
        self._window_size = window_size
        self._filter = capture_filter
        self._configuration = configuration

    def capture(self, window=None):
        window = window or find_iphone_mirroring_window()
        with self._lock:
            self._configure(window)

            def start(callback):
                def received(image, error):
                    if image is None:
                        callback(None, error or "no image was returned")
                        return
                    with native_autorelease_pool():
                        provider = Quartz.CGImageGetDataProvider(image)
                        pixels = bytes(Quartz.CGDataProviderCopyData(provider))
                        value = (
                            Quartz.CGImageGetWidth(image),
                            Quartz.CGImageGetHeight(image),
                            Quartz.CGImageGetBytesPerRow(image),
                            pixels,
                        )
                    callback(value, error)

                screenshot_manager = ScreenCaptureKit.SCScreenshotManager
                screenshot_manager.captureImageWithFilter_configuration_completionHandler_(
                    self._filter,
                    self._configuration,
                    received,
                )

            captured = self._await(start)
        if captured is None:
            raise ScreenRecordingPermissionError(
                "Could not capture iPhone Mirroring. Allow Screen Recording access in System Settings."
            )
        width, height, bytes_per_row, pixels = captured
        return WindowCapture(
            window,
            None,
            PixelFrame(width, height, bytes_per_row, pixels),
        )


_SCREEN_CAPTURE_BACKEND = ScreenCaptureBackend()


def capture_mirroring_window(window=None):
    """Capture iPhone Mirroring through the reusable ScreenCaptureKit backend."""
    return _SCREEN_CAPTURE_BACKEND.capture(window)


def _png_chunk(chunk_type, data):
    payload = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
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
    crop_width = right - left
    crop_height = bottom - top
    if crop_width < 80 or crop_height < frame.height * 0.22:
        raise ReplyGenerationError(
            "The first profile photo was not sufficiently visible for a safe crop."
        )

    scale = min(1.0, max_edge / max(crop_width, crop_height))
    output_width = max(1, round(crop_width * scale))
    output_height = max(1, round(crop_height * scale))
    rows = bytearray()
    for output_y in range(output_height):
        source_y = top + min(crop_height - 1, int(output_y / scale))
        rows.append(0)  # PNG filter type: None
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


def prompt_viewport_png(capture, max_edge=1400):
    """Encode the readable profile viewport for Qwen prompt rescue."""
    frame = capture.frame
    left = round(frame.width * 0.04)
    right = round(frame.width * 0.96)
    top = round(frame.height * 0.07)
    bottom = round(frame.height * 0.91)
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


class HeartIconDetector:
    """Find heart silhouettes without relying on a fixed screen position."""

    GRID_SIZE = 24

    def __init__(self):
        self.expected_fill, self.expected_boundary = self._expected_heart()

    @classmethod
    def _expected_heart(cls):
        raw_fill = set()
        raw_size = 96
        for row in range(raw_size):
            for column in range(raw_size):
                x = ((column + 0.5) / raw_size - 0.5) * 2.7
                y = 1.28 - (row + 0.5) / raw_size * 2.7
                if (x * x + y * y - 1) ** 3 - x * x * y ** 3 <= 0:
                    raw_fill.add((column, row))

        raw_left = min(x for x, _ in raw_fill)
        raw_right = max(x for x, _ in raw_fill)
        raw_top = min(y for _, y in raw_fill)
        raw_bottom = max(y for _, y in raw_fill)
        size = cls.GRID_SIZE
        fill = {
            (
                round((x - raw_left) / (raw_right - raw_left) * (size - 1)),
                round((y - raw_top) / (raw_bottom - raw_top) * (size - 1)),
            )
            for x, y in raw_fill
        }

        boundary = {
            point
            for point in fill
            if any(
                (point[0] + dx, point[1] + dy) not in fill
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
        }
        return fill, boundary

    @staticmethod
    def _near(point, targets, radius=1):
        x, y = point
        return any(
            (x + dx, y + dy) in targets
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        )

    def shape_score(self, points, bounds):
        """Score a connected component as either a filled or outlined heart."""
        left, top, width, height = bounds
        if not points or width <= 0 or height <= 0:
            return 0.0

        observed = set()
        size = self.GRID_SIZE
        for x, y in points:
            column = min(size - 1, int((x - left) / width * size))
            row = min(size - 1, int((y - top) / height * size))
            observed.add((column, row))

        intersection = len(observed & self.expected_fill)
        union = len(observed | self.expected_fill)
        filled_score = intersection / union if union else 0.0

        boundary_precision = sum(
            self._near(point, self.expected_boundary) for point in observed
        ) / len(observed)
        boundary_recall = sum(
            self._near(point, observed) for point in self.expected_boundary
        ) / len(self.expected_boundary)
        boundary_score = (
            2 * boundary_precision * boundary_recall
            / (boundary_precision + boundary_recall)
            if boundary_precision + boundary_recall
            else 0.0
        )

        mirrored = {(size - 1 - x, y) for x, y in observed}
        symmetry_union = len(observed | mirrored)
        symmetry = len(observed & mirrored) / symmetry_union if symmetry_union else 0.0

        bottom = [x for x, y in observed if y >= size * 0.82]
        bottom_centered = bool(bottom) and abs(sum(bottom) / len(bottom) - (size - 1) / 2) < size * 0.16
        top_left = any(x < size * 0.45 and y < size * 0.42 for x, y in observed)
        top_right = any(x > size * 0.55 and y < size * 0.42 for x, y in observed)
        landmarks = (bottom_centered + top_left + top_right) / 3

        geometry = max(filled_score, boundary_score)
        return 0.72 * geometry + 0.18 * symmetry + 0.10 * landmarks

    @staticmethod
    def hinge_button_score(points, bounds):
        """Match Hinge's black circular button and its white heart cutout."""
        left, top, width, height = bounds
        size = len(HINGE_HEART_BUTTON_TEMPLATE)
        dark_counts = [[0 for _ in range(size)] for _ in range(size)]
        cell_totals = [[0 for _ in range(size)] for _ in range(size)]

        for y in range(top, top + height):
            row = min(size - 1, int((y - top) / height * size))
            for x in range(left, left + width):
                column = min(size - 1, int((x - left) / width * size))
                cell_totals[row][column] += 1

        for x, y in points:
            column = min(size - 1, int((x - left) / width * size))
            row = min(size - 1, int((y - top) / height * size))
            dark_counts[row][column] += 1

        matched_weight = 0.0
        total_weight = 0.0
        for row in range(size):
            for column in range(size):
                total = cell_totals[row][column]
                if not total:
                    continue
                observed_dark = dark_counts[row][column] / total >= 0.45
                expected_dark = HINGE_HEART_BUTTON_TEMPLATE[row][column] == "#"

                # Light pixels well inside the circle form the white heart and
                # are the most discriminating part of the control.
                normalized_x = (column + 0.5) / size - 0.5
                normalized_y = (row + 0.5) / size - 0.5
                inside_circle = normalized_x ** 2 + normalized_y ** 2 < 0.19
                if not expected_dark and inside_circle:
                    weight = 5.0
                elif expected_dark:
                    weight = 1.0
                else:
                    weight = 0.4
                total_weight += weight
                if observed_dark == expected_dark:
                    matched_weight += weight

        return matched_weight / total_weight if total_weight else 0.0

    @staticmethod
    def hinge_patch_score(frame, left, top, size):
        """Score a raw image patch, independent of connected dark regions."""
        matched_weight = 0.0
        total_weight = 0.0
        grid_size = len(HINGE_HEART_BUTTON_TEMPLATE)
        for row, template_row in enumerate(HINGE_HEART_BUTTON_TEMPLATE):
            for column, expected in enumerate(template_row):
                normalized_x = (column + 0.5) / grid_size - 0.5
                normalized_y = (row + 0.5) / grid_size - 0.5
                inside_circle = normalized_x ** 2 + normalized_y ** 2 < 0.19

                # Ignore the area outside the button because the underlying
                # profile image may be light or dark.
                if expected == "." and not inside_circle:
                    continue
                weight = 5.0 if expected == "." else 1.0
                x = left + (column + 0.5) / grid_size * size
                y = top + (row + 0.5) / grid_size * size
                observed_dark = HeartIconDetector._is_candidate_color(
                    frame.color_at(x, y),
                    "Hinge",
                )
                total_weight += weight
                if observed_dark == (expected == "#"):
                    matched_weight += weight
        return matched_weight / total_weight if total_weight else 0.0

    def find_hinge_template(
        self,
        frame,
        mask,
        mask_width,
        mask_height,
        x_start,
        y_start,
        expected_button_size,
    ):
        """Slide the Hinge signature over the right side when dark pixels merge with a photo."""
        integral_width = mask_width + 1
        integral = [0] * (integral_width * (mask_height + 1))
        for y in range(mask_height):
            running = 0
            source_row = y * mask_width
            integral_row = (y + 1) * integral_width
            previous_row = y * integral_width
            for x in range(mask_width):
                running += mask[source_row + x]
                integral[integral_row + x + 1] = integral[previous_row + x + 1] + running

        def dark_count(left, top, size):
            right, bottom = left + size, top + size
            return (
                integral[bottom * integral_width + right]
                - integral[top * integral_width + right]
                - integral[bottom * integral_width + left]
                + integral[top * integral_width + left]
            )

        best_score = 0.0
        best_center = None
        scan_left = max(0, int(frame.width * 0.68) - x_start)
        ring_angles = tuple(index * math.pi / 6 for index in range(12))
        internal_white_cells = []
        template_size = len(HINGE_HEART_BUTTON_TEMPLATE)
        for row, template_row in enumerate(HINGE_HEART_BUTTON_TEMPLATE):
            for column, expected in enumerate(template_row):
                nx = (column + 0.5) / template_size - 0.5
                ny = (row + 0.5) / template_size - 0.5
                if expected == "." and nx * nx + ny * ny < 0.19:
                    internal_white_cells.append((column, row))
        probe_step = max(1, len(internal_white_cells) // 12)
        white_probes = internal_white_cells[::probe_step][:12]

        if expected_button_size < 55:
            # Preserve support for 1x/external displays, where screenshots may
            # still contain a larger rendered phone surface.
            minimum_size, maximum_size = 28, 88
        else:
            minimum_size = max(24, round(expected_button_size * 0.82 / 4) * 4)
            maximum_size = round(expected_button_size * 1.18 / 4) * 4
        maximum_size = min(maximum_size, mask_width, mask_height)
        for size in range(minimum_size, maximum_size + 1, 4):
            radius = size * 0.39
            center_offset = size / 2
            for top in range(0, mask_height - size + 1, 4):
                for left in range(scan_left, mask_width - size + 1, 4):
                    density = dark_count(left, top, size) / (size * size)
                    if not 0.42 <= density <= 0.90:
                        continue

                    center_x = left + center_offset
                    center_y = top + center_offset
                    dark_ring_points = sum(
                        bool(mask[
                            min(mask_height - 1, max(0, round(center_y + math.sin(angle) * radius)))
                            * mask_width
                            + min(mask_width - 1, max(0, round(center_x + math.cos(angle) * radius)))
                        ])
                        for angle in ring_angles
                    )
                    if dark_ring_points < 9:
                        continue

                    global_left = x_start + left
                    global_top = y_start + top
                    light_probe_matches = sum(
                        not self._is_candidate_color(
                            frame.color_at(
                                global_left + (column + 0.5) / template_size * size,
                                global_top + (row + 0.5) / template_size * size,
                            ),
                            "Hinge",
                        )
                        for column, row in white_probes
                    )
                    if light_probe_matches < len(white_probes) * 0.58:
                        continue

                    score = self.hinge_patch_score(frame, global_left, global_top, size)
                    if score > best_score:
                        best_score = score
                        best_center = (
                            global_left + center_offset,
                            global_top + center_offset,
                        )

        return best_center, best_score

    @staticmethod
    def _is_candidate_color(color, platform):
        first, green, third = color
        if platform == "Tinder":
            # Tinder's heart is strongly green; Quartz may expose RGB or BGRA.
            return green > 85 and green - first > 22 and green - third > 10
        # Hinge's target is a dark circular button with a white heart cutout.
        return max(color) < 112 and max(color) - min(color) < 55

    def find_all(self, capture, platform):
        frame = capture.frame
        capture_width = frame.width
        capture_height = frame.height

        if platform == "Tinder":
            x_start, x_end = int(capture_width * 0.05), int(capture_width * 0.95)
            y_start, y_end = int(capture_height * 0.42), int(capture_height * 0.96)
            threshold = 0.66
        else:
            # Hinge places the action heart against the lower-right safe-area
            # edge on some profile layouts, so scan all the way to both edges.
            x_start, x_end = int(capture_width * 0.68), capture_width
            y_start, y_end = int(capture_height * 0.08), capture_height
            threshold = 0.84

        mask_width = x_end - x_start
        mask_height = y_end - y_start
        mask = bytearray(mask_width * mask_height)
        for local_y in range(mask_height):
            row_offset = local_y * mask_width
            for local_x in range(mask_width):
                capture_x = x_start + local_x
                capture_y = y_start + local_y
                # Detect at native capture resolution. Downsampling a Retina
                # frame can disconnect Hinge's thin white heart stroke.
                if self._is_candidate_color(frame.color_at(capture_x, capture_y), platform):
                    mask[row_offset + local_x] = 1

        # Connected-component traversal consumes its mask. Preserve the raw
        # pixels for Hinge's sliding-template fallback.
        template_mask = mask[:] if platform == "Hinge" else None

        candidates = []
        # Hinge uses the raw sliding signature below because its black button
        # frequently merges with dark profile imagery. Components remain useful
        # for Tinder's isolated green heart.
        component_height = mask_height if platform == "Tinder" else 0
        for local_y in range(component_height):
            for local_x in range(mask_width):
                start_index = local_y * mask_width + local_x
                if not mask[start_index]:
                    continue

                stack = [(local_x, local_y)]
                mask[start_index] = 0
                points = []
                min_x = max_x = local_x
                min_y = max_y = local_y

                while stack:
                    x, y = stack.pop()
                    points.append((x, y))
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    for dx, dy in (
                        (-1, -1), (0, -1), (1, -1),
                        (-1, 0), (1, 0),
                        (-1, 1), (0, 1), (1, 1),
                    ):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < mask_width and 0 <= ny < mask_height:
                            index = ny * mask_width + nx
                            if mask[index]:
                                mask[index] = 0
                                stack.append((nx, ny))

                width = max_x - min_x + 1
                height = max_y - min_y + 1
                minimum = 15 if platform == "Tinder" else 24
                maximum = 180 if platform == "Tinder" else 150
                aspect = width / height
                valid_aspect = (
                    0.62 <= aspect <= 1.48
                    if platform == "Tinder"
                    else 0.84 <= aspect <= 1.16
                )
                density = len(points) / (width * height)
                if (
                    width < minimum or height < minimum
                    or width > maximum or height > maximum
                    or not valid_aspect
                    or len(points) < width + height
                    or (platform == "Hinge" and not 0.52 <= density <= 0.88)
                ):
                    continue

                bounds = (min_x, min_y, width, height)
                score = (
                    self.shape_score(points, bounds)
                    if platform == "Tinder"
                    else self.hinge_button_score(points, bounds)
                )
                if score < threshold:
                    continue

                center_x = x_start + min_x + width / 2
                center_y = y_start + min_y + height / 2
                if platform == "Tinder":
                    # Prefer the familiar lower-center action row when shapes tie.
                    position_bonus = max(
                        0,
                        0.08 - abs(center_x / capture_width - 0.5) * 0.08,
                    )
                    score += position_bonus
                else:
                    # Prefer a fully visible heart near the vertical center.
                    score += max(0, 0.04 - abs(center_y / capture_height - 0.5) * 0.04)
                candidates.append((score, center_x, center_y, bounds))

        if platform == "Hinge":
            template_center, template_score = self.find_hinge_template(
                frame,
                template_mask,
                mask_width,
                mask_height,
                x_start,
                y_start,
                36 * capture.scale_x,
            )
            if template_center is not None and template_score >= threshold:
                candidates.append((
                    template_score,
                    template_center[0],
                    template_center[1],
                    None,
                ))

        if not candidates:
            return []
        candidates.sort(reverse=True)
        results = []
        for score, center_x, center_y, _ in candidates:
            point = capture.desktop_point(center_x, center_y)
            if any(
                abs(point[0] - current_point[0]) < 8
                and abs(point[1] - current_point[1]) < 8
                for current_point, _ in results
            ):
                continue
            results.append((point, min(score, 1.0)))
        return results

    def find(self, capture, platform):
        candidates = self.find_all(capture, platform)
        return candidates[0] if candidates else (None, 0.0)


def _normalize_text(text):
    return re.sub(r"[^a-z]+", " ", text.casefold()).strip()


def is_safe_iphone_action_point(capture, point, *, bottom_limit=0.86):
    """Reject action coordinates near iPhone's status and home gesture areas."""
    if point is None:
        return False
    window = capture.window
    relative_x = (point[0] - window.left) / window.width
    relative_y = (point[1] - window.top) / window.height
    return 0.04 <= relative_x <= 0.96 and 0.12 <= relative_y <= bottom_limit


SEND_SETTLE_TIMEOUT = 1.1
SEND_POLL_INTERVAL = 0.15


def find_send_priority_like(capture, lines=None):
    """Use native macOS OCR to find Hinge's current confirmation button."""
    if lines is None:
        lines = recognize_text(capture, Vision)
    candidates = []
    for line in lines:
        normalized = _normalize_text(line.text)
        words = set(normalized.split())
        if {"send", "like"}.issubset(words):
            score = line.confidence + (0.25 if "priority" in words else 0)
            candidates.append((score, line.center, line.text))
    if not candidates:
        return None, 0.0, ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, point, text = candidates[0]
    return point, min(score, 1.0), text


def find_hinge_skip_x(capture):
    """Verify Hinge's floating lower-left X before using it to skip a profile."""
    frame = capture.frame
    expected_x = frame.width * 0.135
    radius = max(8.0, frame.width * 0.026)

    def brightness(x, y):
        color = frame.color_at(x, y)
        return sum(color) / max(1, len(color))

    best = (0.0, None)
    search_radius = max(3, round(radius * 0.45))
    step = max(1, round(radius * 0.18))
    offsets = (-0.68, -0.50, -0.32, 0.0, 0.32, 0.50, 0.68)
    # The navigation bar collapses while scrolling, moving the floating X
    # from roughly 85% to 92% of the mirrored viewport.
    for center_y in range(round(frame.height * 0.79), round(frame.height * 0.94), step):
        for center_x in range(
            round(expected_x - search_radius),
            round(expected_x + search_radius) + 1,
            step,
        ):
            diagonal = []
            for fraction in offsets:
                distance = radius * fraction
                diagonal.append(brightness(center_x + distance, center_y + distance) < 115)
                diagonal.append(brightness(center_x + distance, center_y - distance) < 115)
            surround = [
                brightness(center_x + radius, center_y) > 185,
                brightness(center_x - radius, center_y) > 185,
                brightness(center_x, center_y + radius) > 185,
                brightness(center_x, center_y - radius) > 185,
            ]
            score = 0.75 * (sum(diagonal) / len(diagonal)) + 0.25 * (
                sum(surround) / len(surround)
            )
            if score > best[0]:
                best = (score, (center_x, center_y))
    if best[1] is None or best[0] < 0.70:
        return None, best[0]
    return capture.desktop_point(*best[1]), min(1.0, best[0])


class MacClipboard:
    """Temporarily replace the macOS clipboard while preserving every data type."""

    def __init__(self, appkit_module):
        self.appkit = appkit_module

    def _pasteboard(self):
        if self.appkit is None:
            raise RuntimeError("Clipboard access requires pyobjc-framework-Cocoa.")
        return self.appkit.NSPasteboard.generalPasteboard()

    def snapshot(self):
        saved = []
        for item in self._pasteboard().pasteboardItems() or []:
            values = []
            for pasteboard_type in item.types() or []:
                data = item.dataForType_(pasteboard_type)
                if data is not None:
                    values.append((pasteboard_type, data))
            saved.append(values)
        return saved

    def set_text(self, value):
        pasteboard = self._pasteboard()
        pasteboard.clearContents()
        pasteboard.setString_forType_(value, self.appkit.NSPasteboardTypeString)

    def restore(self, saved):
        pasteboard = self._pasteboard()
        pasteboard.clearContents()
        items = []
        for values in saved:
            item = self.appkit.NSPasteboardItem.alloc().init()
            for pasteboard_type, data in values:
                item.setData_forType_(data, pasteboard_type)
            items.append(item)
        if items:
            pasteboard.writeObjects_(items)


class ScoopUpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Scoop UP — Automatic Detection")
        self.root.geometry("500x585")
        self.root.resizable(True, False)

        self.mouse = Controller()
        self.keyboard = KeyboardController()
        self.clipboard = MacClipboard(AppKit)
        self.heart_detector = HeartIconDetector()
        self.is_running = False
        self.total_rotations = 20

        tk.Label(
            root,
            text="Automatic iPhone Mirroring Detection",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=(14, 8))

        tk.Label(root, text="Dating app:").pack()
        self.platform_var = tk.StringVar(value="Hinge")
        tk.OptionMenu(root, self.platform_var, "Hinge", "Tinder").pack()

        tk.Label(root, text="Workflow:").pack(pady=(8, 0))
        self.workflow_var = tk.StringVar(value="Auto Like")
        tk.OptionMenu(root, self.workflow_var, "Auto Like", "Prompt Reply").pack()

        tk.Label(root, text="Reply engine:").pack(pady=(8, 0))
        self.engine_var = tk.StringVar(value="Local — Free")
        tk.OptionMenu(root, self.engine_var, "Local — Free", "OpenAI API").pack()

        tk.Label(root, text="Reply tone:").pack(pady=(8, 0))
        self.tone_var = tk.StringVar(value="Playful & clean")
        tk.OptionMenu(root, self.tone_var, *TONE_INSTRUCTIONS.keys()).pack()

        tk.Label(root, text="Fallback pickup line (optional):").pack(pady=(8, 0))
        self.fallback_line_entry = tk.Entry(root, width=52, justify="center")
        self.fallback_line_entry.pack()
        tk.Label(
            root,
            text="Leave blank to generate a line from the first photo locally.",
            font=("Helvetica", 10),
        ).pack()

        tk.Label(root, text="Number of rotations:").pack(pady=(8, 0))
        self.rotations_entry = tk.Entry(root, width=12, justify="center")
        self.rotations_entry.insert(0, "20")
        self.rotations_entry.pack()

        button_frame = tk.Frame(root)
        button_frame.pack(pady=12)
        self.start_button = tk.Button(button_frame, text="Start", command=self.on_start, width=10)
        self.start_button.grid(row=0, column=0, padx=5)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop, width=10)
        self.stop_button.grid(row=0, column=1, padx=5)
        self.restart_button = tk.Button(button_frame, text="Restart", command=self.restart, width=10)
        self.restart_button.grid(row=0, column=2, padx=5)

        self.status_label = tk.Label(
            root,
            text="Open iPhone Mirroring, choose an app, then press Start.",
            wraplength=390,
            justify="center",
        )
        self.status_label.pack(padx=12)

        self.settings_button = tk.Button(
            root,
            text="Open Screen Recording Settings",
            command=self.open_screen_recording_settings,
        )
        # This button is intentionally hidden until capture permission fails.
        self.root.bind("<Escape>", lambda _event: self.stop())

    def render_status(self, message, show_settings=False):
        self.status_label.config(text=message)
        if show_settings:
            if not self.settings_button.winfo_manager():
                self.settings_button.pack(pady=(12, 0))
        else:
            self.settings_button.pack_forget()

    def set_status(self, message, show_settings=False):
        self.root.after(0, lambda: self.render_status(message, show_settings))

    def open_screen_recording_settings(self):
        """Open Privacy & Security directly at Screen Recording permissions."""
        if AppKit is None:
            self.render_status("Unable to open System Settings automatically.", True)
            return
        url = AppKit.NSURL.URLWithString_(SCREEN_RECORDING_SETTINGS_URL)
        opened = AppKit.NSWorkspace.sharedWorkspace().openURL_(url)
        if opened:
            self.render_status(
                "Enable access for Terminal or Python, then restart The Scoop UP.",
                True,
            )
        else:
            self.render_status(
                "Could not open System Settings. Open Privacy & Security → Screen Recording manually.",
                True,
            )

    def interruptible_wait(self, seconds):
        deadline = time.monotonic() + seconds
        while self.is_running and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def fresh_capture(self):
        # Window position and size are deliberately resolved again after every
        # click because the user may move or resize iPhone Mirroring.
        return capture_mirroring_window(find_iphone_mirroring_window())

    def click_repeatedly(self, point, count=3, interval=0.12):
        """Send a short click burst to a target that was positively detected."""
        self.mouse.position = point
        for click_index in range(count):
            if not self.is_running:
                break
            self.mouse.click(Button.left, 1)
            if click_index < count - 1:
                self.interruptible_wait(interval)

    def click_once(self, point):
        if not self.is_running:
            return
        self.mouse.position = point
        self.mouse.click(Button.left, 1)

    def scroll_profile(self, direction):
        """Focus iPhone Mirroring and send pixel-based trackpad-style scrolling."""
        window = find_iphone_mirroring_window()
        if AppKit is None or Quartz is None:
            raise RuntimeError("Native macOS scrolling support is unavailable.")
        mirrored_app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(
            window.owner_pid
        )
        if mirrored_app is not None:
            mirrored_app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
        self.mouse.position = (
            round(window.left + window.width * 0.52),
            round(window.top + window.height * 0.55),
        )
        # Two smaller pixel events produce roughly 70% viewport overlap on
        # the mirrored phone, keeping a prompt heading and its heart visible
        # together often enough for safe association.
        small_scroll = direction.endswith("_small")
        base_direction = direction.removesuffix("_small")
        delta = (32 if small_scroll else 140) * (
            1 if base_direction == "up" else -1
        )
        event_count = 1 if small_scroll else 2
        for _ in range(event_count):
            if not self.is_running:
                break
            event = Quartz.CGEventCreateScrollWheelEvent(
                None,
                Quartz.kCGScrollEventUnitPixel,
                1,
                delta,
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.08)

    def make_profile_scanner(self, generator=None):
        vision_rescue = None
        if hasattr(generator, "detect_written_prompt"):
            def vision_rescue(capture, lines, hearts, viewport_index):
                try:
                    detected = generator.detect_written_prompt(
                        prompt_viewport_png(capture)
                    )
                except (ReplyGenerationError, OSError, RuntimeError):
                    return None
                if detected is None:
                    return None
                prompt_text, answer_text = detected
                return recover_vision_prompt_target(
                    prompt_text,
                    answer_text,
                    lines,
                    hearts,
                    viewport_index,
                    capture.window,
                )

        return ProfileScanner(
            capture=self.fresh_capture,
            ocr=lambda capture: recognize_text(capture, Vision),
            find_hearts=lambda capture: self.heart_detector.find_all(capture, "Hinge"),
            scroll=self.scroll_profile,
            wait=self.interruptible_wait,
            is_running=lambda: self.is_running,
            progress=self.set_status,
            vision_rescue=vision_rescue,
            fast_jump_viewports=1,
        )

    def clear_reply_field(self):
        # Hinge preserves unsent drafts per profile. Always replace the
        # composer contents so a stale reply can never survive into a new
        # generated prompt response.
        self.keyboard.press(Key.cmd)
        self.keyboard.press("a")
        self.keyboard.release("a")
        self.keyboard.release(Key.cmd)
        self.interruptible_wait(0.15)
        self.keyboard.press(Key.backspace)
        self.keyboard.release(Key.backspace)
        self.interruptible_wait(0.15)

    def paste_reply(self, reply):
        saved = self.clipboard.snapshot()
        try:
            self.clear_reply_field()
            self.clipboard.set_text(reply)
            # Give iPhone Mirroring time to observe the new pasteboard change
            # before requesting paste; otherwise it can reuse the prior value.
            self.interruptible_wait(0.3)
            self.keyboard.press(Key.cmd)
            self.keyboard.press("v")
            self.keyboard.release("v")
            self.keyboard.release(Key.cmd)
            # iPhone Mirroring consumes pasteboard data asynchronously. Keep
            # our temporary clipboard value alive until the phone has read it.
            self.interruptible_wait(0.8)
        finally:
            self.clipboard.restore(saved)

    def type_reply(self, reply):
        if not reply.isascii():
            raise ProfileScanError(
                "Direct typing fallback requires an ASCII reply."
            )
        self.clear_reply_field()
        for character in reply:
            if not self.is_running:
                return
            self.keyboard.type(character)
            time.sleep(0.006)

    def verify_reply_entry(
        self,
        reply,
        comment_point=None,
        window_height=None,
        attempts=3,
    ):
        """Require OCR evidence of the pasted reply before permitting a send."""
        last_capture = None
        last_lines = []
        for _ in range(attempts):
            if not self.is_running:
                break
            last_capture = self.fresh_capture()
            last_lines = recognize_text(last_capture, Vision)
            verified = (
                reply_is_visible_near(
                    last_lines,
                    reply,
                    comment_point,
                    window_height,
                )
                if comment_point is not None and window_height is not None
                else reply_is_visible(last_lines, reply)
            )
            if verified:
                return last_capture, last_lines
            if _ < attempts - 1:
                self.interruptible_wait(0.15)
        return last_capture, last_lines

    def find_send_for_dialog(self, capture, lines, comment_point):
        point, score, label = find_send_priority_like(capture, lines)
        if (
            point is not None
            and score >= 0.55
            and is_safe_iphone_action_point(capture, point)
        ):
            return point, score, label
        # Vision sometimes garbles the pale button label. Once the matching
        # dialog and composer text are independently verified, its control is
        # reliably the wide button immediately below the composer.
        window = capture.window
        fallback = (
            window.left + window.width * 0.62,
            comment_point[1] + window.height * 0.085,
        )
        if is_safe_iphone_action_point(capture, fallback):
            return fallback, 0.60, "Send Like (verified layout)"
        return None, 0.0, ""

    def position_open_dialog_safely(self, selected=None, attempts=7):
        """Lift the expanded composer until Send Like is in the top 85%."""
        for attempt in range(1, attempts + 1):
            capture = self.fresh_capture()
            lines = recognize_text(capture, Vision)
            _, comment_point, _ = find_text_target(
                lines,
                ("Add a comment", "Write a comment", "Say something"),
                min_confidence=0.25,
            )
            raw_send, raw_score, _ = find_send_priority_like(capture, lines)
            if comment_point is None and raw_send is not None and raw_score >= 0.55:
                comment_point = (
                    capture.window.left + capture.window.width * 0.50,
                    raw_send[1] - capture.window.height * 0.085,
                )
            candidate_send = raw_send
            candidate_score = raw_score
            if comment_point is not None and candidate_send is None:
                candidate_send = (
                    capture.window.left + capture.window.width * 0.62,
                    comment_point[1] + capture.window.height * 0.085,
                )
                candidate_score = 0.60
            if (
                comment_point is not None
                and candidate_send is not None
                and candidate_score >= 0.55
                and is_safe_iphone_action_point(
                    capture,
                    candidate_send,
                    bottom_limit=0.85,
                )
            ):
                return capture, lines, comment_point
            if attempt < attempts:
                self.set_status(
                    f"Moving Send Like into the top 85% ({attempt}/{attempts})..."
                )
                self.scroll_profile("down_small")
                self.interruptible_wait(0.18)
        raise ProfileScanError(
            "Send Like could not be moved into the top 85%; nothing was clicked."
        )

    def enter_reply(self, comment_point, reply, window_height, attempts=3):
        """Focus, replace, paste, and composer-verify with bounded retries."""
        last_capture = None
        last_lines = []
        for attempt in range(1, attempts + 1):
            self.set_status(f"Entering generated reply ({attempt}/{attempts})...")
            self.click_once(comment_point)
            self.interruptible_wait(0.2)
            if reply.isascii():
                self.type_reply(reply)
            else:
                self.paste_reply(reply)
            self.interruptible_wait(0.12)
            last_capture, last_lines = self.verify_reply_entry(
                reply,
                comment_point,
                window_height,
            )
            if reply_is_visible_near(
                last_lines,
                reply,
                comment_point,
                window_height,
            ):
                return last_capture, last_lines
        return last_capture, last_lines

    def wait_for_comment_field(self, attempts=6):
        """Wait for Hinge's like sheet, then locate its comment composer."""
        for attempt in range(attempts):
            if not self.is_running:
                break
            self.set_status(
                f"Opening Add a comment field ({attempt + 1}/{attempts})..."
            )
            capture = self.fresh_capture()
            lines = recognize_text(capture, Vision)
            _, comment_point, _ = find_text_target(
                lines,
                ("Add a comment", "Write a comment", "Say something"),
                # Placeholder text is pale gray and can receive lower Vision
                # confidence than the surrounding prompt and action buttons.
                min_confidence=0.25,
            )
            if comment_point is not None:
                return comment_point

            send_score, send_point, _ = find_text_target(
                lines,
                ("Send Priority Like", "Send Like"),
            )
            if send_point is not None and send_score >= 0.55:
                # The send control positively identifies the Hinge like sheet.
                # Its composer occupies the full-width box immediately above
                # that control, even when its pale placeholder is not OCR'd.
                window = capture.window
                composer_point = (
                    window.left + window.width * 0.50,
                    # In the compact Hinge sheet the composer is immediately
                    # above Send.  The previous 18% offset landed in the
                    # suggestion chips, so text could paste successfully but
                    # verification watched the wrong part of the screen.
                    send_point[1] - window.height * 0.085,
                )
                if (
                    window.left < composer_point[0] < window.left + window.width
                    and window.top < composer_point[1] < window.top + window.height
                ):
                    return composer_point
            self.interruptible_wait(0.35)
        return None

    def wait_for_matching_prompt_dialog(self, selected, attempts=3):
        """Tolerate a transient OCR miss after Hinge expands the prompt."""
        last_capture = None
        last_lines = []
        for _ in range(attempts):
            last_capture = self.fresh_capture()
            last_lines = recognize_text(last_capture, Vision)
            if prompt_is_visible(last_lines, selected):
                return last_capture, last_lines
            self.interruptible_wait(0.25)
        return last_capture, last_lines

    def open_prompt_dialog(self, scanner, selected, relocated, attempts=2):
        """Click a verified prompt heart, then hand off to Send Like discovery."""
        current_target = relocated
        self.set_status("Opening selected prompt...")
        # iPhone Mirroring occasionally drops individual taps. The heart
        # target was freshly confirmed, so use the same short burst as the
        # proven Auto Like workflow before re-reading the resulting state.
        capture = self.fresh_capture()
        if not is_safe_iphone_action_point(
            capture,
            current_target.heart_point,
            bottom_limit=0.72,
        ):
            self.set_status(
                "Prompt heart is in the bottom safe-zone margin; moving it upward..."
            )
            current_target = scanner.center_target(selected, current_target)
            capture = self.fresh_capture()
            if not is_safe_iphone_action_point(
                capture,
                current_target.heart_point,
                bottom_limit=0.72,
            ):
                raise ProfileScanError(
                    "The prompt heart could not be moved out of the iPhone home area; "
                    "nothing was clicked."
                )
        for click_index in range(3):
            self.click_once(current_target.heart_point)
            if click_index < 2:
                self.interruptible_wait(0.15)
        self.interruptible_wait(0.55)
        if not self.is_running:
            return None
        comment_point = self.wait_for_comment_field(attempts=2)
        # The composer can begin below the viewport. Do not re-detect the
        # original prompt underneath an opening like sheet; hand off to
        # position_open_dialog_safely(), which scrolls until Send Like appears.
        return comment_point

    def find_send_like_before(self, deadline):
        """Run one bounded OCR scan without leaving timed-out work behind."""
        capture = None
        with native_autorelease_pool():
            try:
                capture = self.fresh_capture()
                result = find_send_priority_like(capture)
            finally:
                capture = None
        if time.monotonic() > deadline:
            return None, 0.0, ""
        return result

    def find_heart_before(self, platform, deadline):
        """Run one heart scan at a time so expired scans cannot accumulate."""
        capture = None
        with native_autorelease_pool():
            try:
                capture = self.fresh_capture()
                point, score = self.heart_detector.find(capture, platform)
            finally:
                capture = None
        if time.monotonic() > deadline:
            return None, 0.0, True
        return point, score, False

    def run_auto_like(self, platform):
        completed = 0
        missing_hearts = 0
        missing_send_buttons = 0

        try:
            for cycle in range(1, self.total_rotations + 1):
                if not self.is_running:
                    break

                heart_point, heart_score, heart_timed_out = self.find_heart_before(
                    platform,
                    time.monotonic() + 2,
                )
                if heart_point is None:
                    missing_hearts += 1
                    if heart_timed_out:
                        self.set_status(
                            f"Cycle {cycle}: heart scan timed out; retrying next cycle."
                        )
                        continue
                    if platform == "Hinge":
                        # The confirmation UI may already be open from a prior
                        # cycle, so recover by trying Send Like directly.
                        self.set_status(
                            f"Cycle {cycle}: no heart after 2 seconds; trying Send Like..."
                        )
                        send_point, send_score, detected_text = self.find_send_like_before(
                            time.monotonic() + 2
                        )
                        if send_point is not None:
                            self.click_repeatedly(send_point)
                            completed += 1
                            self.set_status(
                                f"Cycle {cycle}: recovered with {detected_text} "
                                f"({send_score:.0%}) and clicked 3 times."
                            )
                            self.interruptible_wait(2)
                        else:
                            missing_send_buttons += 1
                            self.set_status(
                                f"Cycle {cycle}: heart and Send Like not detected; skipping."
                            )
                    else:
                        self.set_status(
                            f"Cycle {cycle}: heart not detected in 2 seconds; skipping."
                        )
                    continue

                # Detection and coordinates came from this fresh frame.
                self.click_repeatedly(heart_point)
                self.set_status(
                    f"Cycle {cycle}: heart detected ({heart_score:.0%}) and clicked 3 times."
                )

                if platform == "Hinge":
                    # Give Hinge a full second to render its confirmation UI,
                    # then allow Send Like detection its own two-second window.
                    self.interruptible_wait(1)
                    if not self.is_running:
                        break

                    # Capture and OCR again after the heart click. No previous
                    # button coordinates are reused.
                    self.set_status(f"Cycle {cycle}: looking for Send Priority Like...")
                    send_point, send_score, detected_text = self.find_send_like_before(
                        time.monotonic() + 2
                    )
                    if send_point is None:
                        missing_send_buttons += 1
                        self.set_status(
                            f"Cycle {cycle}: Send Priority Like not detected in 2 seconds; skipping."
                        )
                        continue

                    self.click_repeatedly(send_point)
                    self.set_status(
                        f"Cycle {cycle}: {detected_text} detected ({send_score:.0%}) "
                        "and clicked 3 times."
                    )

                completed += 1
                self.interruptible_wait(2)

            if self.is_running:
                summary = (
                    f"Done: {completed} completed, {missing_hearts} missing hearts"
                )
                if platform == "Hinge":
                    summary += f", {missing_send_buttons} missing send buttons"
                self.set_status(summary + ".")
            else:
                self.set_status("Stopped.")
        except ScreenRecordingPermissionError as error:
            self.set_status(str(error), show_settings=True)
        except Exception as error:
            self.set_status(str(error))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def _start_reply_generation(self, generator, prompts, tone):
        box = {"reply": None, "error": None}

        def worker():
            try:
                box["reply"] = generator.generate(prompts, tone)
            except Exception as error:
                box["error"] = error

        thread = threading.Thread(
            target=worker,
            daemon=True,
            name="scoop-reply-generate",
        )
        thread.start()
        return thread, box

    def _finish_reply_generation(self, generation, selected):
        thread, box = generation
        while thread.is_alive() and self.is_running:
            thread.join(0.1)
        if thread.is_alive():
            raise ProfileScanError(
                "Prompt reply stopped before a reply was generated."
            )
        if box["error"] is not None:
            raise box["error"]
        generated = box["reply"]
        if generated is None:
            raise ProfileScanError("Reply generation did not return a reply.")
        if generated.prompt_id != selected.prompt_id:
            raise ProfileScanError(
                "The generated reply no longer matched the opened prompt; nothing was entered or sent."
            )
        return generated

    def _classify_prompt_send_outcome(
        self,
        capture,
        lines,
        selected,
        reply,
        comment_point,
        window_height,
    ):
        send_point, _, _ = find_send_priority_like(capture, lines)
        reply_visible = reply_is_visible_near(
            lines,
            reply,
            comment_point,
            window_height,
        )
        prompt_visible = prompt_is_visible(lines, selected)
        if send_point is None:
            # Hinge closes the composer after a successful send. Leftover OCR
            # near the old composer, or the prompt card behind it, must not
            # veto that close and abort the rest of the rotation batch.
            return "succeeded"
        if reply_visible and prompt_visible:
            return "retry"
        return "uncertain"

    def _observe_after_send_click(self, classify, timeout=SEND_SETTLE_TIMEOUT):
        """Poll until Send settles or the timeout is reached."""
        polls = max(1, math.ceil(timeout / SEND_POLL_INTERVAL))
        last_capture = None
        last_lines = []
        last_outcome = "retry"
        confirmed = False
        for poll_index in range(polls):
            if not self.is_running:
                break
            last_capture = self.fresh_capture()
            use_accurate = poll_index == polls - 1
            last_lines = recognize_text(
                last_capture,
                Vision,
                accurate=use_accurate,
            )
            last_outcome = classify(last_capture, last_lines)
            if last_outcome != "retry" or use_accurate:
                if not use_accurate:
                    last_lines = recognize_text(last_capture, Vision, accurate=True)
                    last_outcome = classify(last_capture, last_lines)
                confirmed = True
                if last_outcome != "retry":
                    return last_capture, last_lines, last_outcome
            if poll_index < polls - 1:
                self.interruptible_wait(SEND_POLL_INTERVAL)
        if last_capture is not None and not confirmed:
            last_lines = recognize_text(last_capture, Vision, accurate=True)
            last_outcome = classify(last_capture, last_lines)
        return last_capture, last_lines, last_outcome

    def _process_prompt_reply_profile(
        self,
        scanner,
        generator,
        tone,
        cycle,
        ensure_top,
    ):
        self._prompt_failure_can_skip = True
        self._prompt_stage = "scan"
        self.set_status(f"Profile {cycle}: looking for the first written prompt...")
        scan = scanner.scan(ensure_top=ensure_top)
        selected = scan.prompts[0]

        self._prompt_stage = "generate"
        self.set_status(
            f"Profile {cycle}: generating from {len(scan.prompts)} prompts "
            "while preparing the prompt..."
        )
        generation = self._start_reply_generation(generator, scan.prompts, tone)

        if selected.confidence < 0.55:
            raise ProfileScanError(
                "The selected prompt target was below the safe confidence threshold."
            )

        self._prompt_stage = "open_dialog"
        self.set_status(f"Profile {cycle}: prompt heart detected; opening it now...")
        self.open_prompt_dialog(scanner, selected, selected)
        if not self.is_running:
            raise ProfileScanError(
                "Prompt reply stopped before the like dialog could be positioned."
            )
        # Composer/Send Like often start below the viewport. Scroll them into
        # place before requiring OCR of the opened prompt.
        dialog_capture, dialog_lines, comment_point = self.position_open_dialog_safely(
            selected
        )
        if not prompt_is_visible(dialog_lines, selected):
            _capture, dialog_lines = self.wait_for_matching_prompt_dialog(selected)
            if not prompt_is_visible(dialog_lines, selected):
                raise ProfileScanError(
                    "The opened Hinge dialog did not match the generated prompt; nothing was entered or sent."
                )

        self._prompt_stage = "generate"
        self.set_status(f"Profile {cycle}: Send Like is positioned; finishing the reply...")
        generated = self._finish_reply_generation(generation, selected)
        self._current_generated_reply = generated.reply
        self.set_status(
            f"Profile {cycle}: replying to {selected.prompt!r} with {generated.reply!r}"
        )

        self._prompt_stage = "enter_reply"
        entered_capture, entered_lines = self.enter_reply(
            comment_point,
            generated.reply,
            dialog_capture.window.height,
        )
        if entered_capture is None or not reply_is_visible_near(
            entered_lines,
            generated.reply,
            comment_point,
            dialog_capture.window.height,
        ):
            raise ProfileScanError(
                "The entered reply could not be verified; nothing was sent."
            )

        self._prompt_stage = "send"
        send_succeeded = False
        current_capture = entered_capture
        current_lines = entered_lines
        window_height = dialog_capture.window.height

        def classify_send(capture, lines):
            return self._classify_prompt_send_outcome(
                capture,
                lines,
                selected,
                generated.reply,
                comment_point,
                window_height,
            )

        for send_attempt in range(1, 4):
            if send_attempt > 1:
                current_capture = self.fresh_capture()
                current_lines = recognize_text(current_capture, Vision)
                if not reply_is_visible_near(
                    current_lines,
                    generated.reply,
                    comment_point,
                    window_height,
                ):
                    raise ProfileScanError(
                        "The reply changed while retrying Send; stopped without another click."
                    )
                if not prompt_is_visible(current_lines, selected):
                    raise ProfileScanError(
                        "The prompt changed while retrying Send; stopped without another click."
                    )

            send_point, send_score, detected_text = self.find_send_for_dialog(
                current_capture,
                current_lines,
                comment_point,
            )
            if send_point is None or send_score < 0.55:
                raise ProfileScanError(
                    "The Send Like control was not confidently detected in the safe screen area; nothing was sent."
                )
            self.set_status(
                f"Profile {cycle}: verified reply; clicking {detected_text} "
                f"({send_attempt}/3)..."
            )
            self._prompt_failure_can_skip = False
            self.click_once(send_point)
            _after_capture, _after_lines, outcome = self._observe_after_send_click(
                classify_send
            )
            if outcome == "succeeded":
                send_succeeded = True
                break
            if outcome == "unexpected":
                raise ProfileScanError(
                    "The send dialog changed unexpectedly; stopped without another click."
                )
            if outcome == "uncertain":
                raise ProfileScanError(
                    "The send state became uncertain; stopped without another click."
                )
            # The same verified dialog remains, so it is safe to dismiss and
            # skip this profile if all bounded Send attempts fail.
            self._prompt_failure_can_skip = True

        if not send_succeeded:
            raise ProfileScanError(
                "Send Like did not respond after 3 verified click attempts."
            )
        return generated.reply

    def log_prompt_failure(
        self,
        batch_id,
        cycle,
        stage,
        error,
        skipped,
        recovery=None,
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch_id,
            "rotation": cycle,
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "skipped": bool(skipped),
            "recovery": recovery,
        }
        try:
            PROMPT_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with PROMPT_FAILURE_LOG.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return record

    def fallback_regular_like(
        self,
        scanner,
        cycle,
        custom_pickup_line="",
        photo_generator=None,
        tone="Playful & clean",
    ):
        """Recover a failed prompt with a verified clean pickup-line comment."""
        pickup_line = (
            validate_fallback_pickup_line(custom_pickup_line)
            if custom_pickup_line.strip()
            else None
        )
        self.set_status(
            f"Profile {cycle}: prompt failed; preparing a photo-grounded fallback..."
        )
        capture = self.fresh_capture()
        send_point, _, _ = find_send_priority_like(capture)
        if send_point is not None:
            # Remove any partially entered comment before leaving the failed
            # inline composer. This prevents a stale draft from being sent by
            # the regular-like fallback.
            comment_point = (
                capture.window.left + capture.window.width * 0.50,
                send_point[1] - capture.window.height * 0.085,
            )
            self.click_once(comment_point)
            self.interruptible_wait(0.25)
            self.clear_reply_field()

        scanner.scroll_to_top()
        capture = self.fresh_capture()
        hearts = [
            (point, score)
            for point, score in self.heart_detector.find_all(capture, "Hinge")
            if 0.18
            <= (point[1] - capture.window.top) / capture.window.height
            <= 0.72
            and score >= 0.55
        ]
        if not hearts:
            return False
        heart_point, _ = min(hearts, key=lambda item: item[0][1])

        if pickup_line is None and hasattr(photo_generator, "generate_photo_pickup_line"):
            try:
                self.set_status(
                    f"Profile {cycle}: generating a line from the first photo locally..."
                )
                photo_png = first_profile_photo_png(capture, heart_point)
                pickup_line = photo_generator.generate_photo_pickup_line(photo_png, tone)
                self.set_status(
                    f"Profile {cycle}: photo-grounded fallback ready: {pickup_line!r}"
                )
            except (ReplyGenerationError, OSError, RuntimeError) as error:
                pickup_line = random_pickup_line()
                self.set_status(
                    f"Profile {cycle}: photo fallback unavailable ({error}); "
                    "using a safe built-in line."
                )
        elif pickup_line is None:
            pickup_line = random_pickup_line()

        # Image generation can take several seconds. Re-capture and require the
        # same viewport before using any previously observed profile state.
        current_capture = self.fresh_capture()
        if viewport_similarity(
            [],
            getattr(capture, "frame", None),
            [],
            getattr(current_capture, "frame", None),
        ) < 0.90:
            return False
        current_hearts = [
            (point, score)
            for point, score in self.heart_detector.find_all(current_capture, "Hinge")
            if 0.18
            <= (point[1] - current_capture.window.top) / current_capture.window.height
            <= 0.72
            and score >= 0.55
        ]
        if not current_hearts:
            return False
        heart_point, _ = min(current_hearts, key=lambda item: item[0][1])
        for click_index in range(3):
            self.click_once(heart_point)
            if click_index < 2:
                self.interruptible_wait(0.12)
        self.interruptible_wait(0.55)

        comment_point = self.wait_for_comment_field(attempts=3)
        if comment_point is None:
            return False
        try:
            dialog_capture, _, comment_point = self.position_open_dialog_safely()
        except ProfileScanError:
            return False
        entered_capture, entered_lines = self.enter_reply(
            comment_point,
            pickup_line,
            dialog_capture.window.height,
            attempts=2,
        )
        if entered_capture is None or not reply_is_visible_near(
            entered_lines,
            pickup_line,
            comment_point,
            dialog_capture.window.height,
        ):
            return False

        for send_attempt in range(1, 4):
            dialog_capture = self.fresh_capture()
            dialog_lines = recognize_text(dialog_capture, Vision)
            if not reply_is_visible_near(
                dialog_lines,
                pickup_line,
                comment_point,
                dialog_capture.window.height,
            ):
                return False
            send_point, send_score, detected_text = self.find_send_for_dialog(
                dialog_capture,
                dialog_lines,
                comment_point,
            )
            if send_point is None or send_score < 0.55:
                return False
            self.set_status(
                f"Profile {cycle}: sending pickup-line {detected_text} "
                f"({send_attempt}/3)..."
            )
            self.click_once(send_point)
            before_lines = dialog_lines
            before_frame = getattr(dialog_capture, "frame", None)

            def classify_fallback(capture, lines):
                after_send, _, _ = find_send_priority_like(capture, lines)
                if after_send is None:
                    similarity = viewport_similarity(
                        before_lines,
                        before_frame,
                        lines,
                        getattr(capture, "frame", None),
                    )
                    return "succeeded" if similarity < 0.92 else "failed"
                return "retry"

            _after_capture, _after_lines, outcome = self._observe_after_send_click(
                classify_fallback
            )
            if outcome == "succeeded":
                self.interruptible_wait(0.25)
                return True
            if outcome == "failed":
                return False
        return False

    def skip_current_profile(self, attempts=3):
        """Dismiss an unsent sheet, verify Hinge's X, and confirm profile change."""
        for attempt in range(1, attempts + 1):
            if not self.is_running:
                return False
            capture = self.fresh_capture()
            lines = recognize_text(capture, Vision)
            send_point, _, _ = find_send_priority_like(capture, lines)
            skip_point, skip_score = find_hinge_skip_x(capture)
            if send_point is not None or skip_point is None or skip_score < 0.70:
                self.set_status(
                    f"Dismissing leftover like sheet before skip ({attempt}/{attempts})..."
                )
                self.keyboard.press(Key.esc)
                self.keyboard.release(Key.esc)
                self.interruptible_wait(0.5)
                capture = self.fresh_capture()
                lines = recognize_text(capture, Vision)
                skip_point, skip_score = find_hinge_skip_x(capture)
            if skip_point is None or skip_score < 0.70:
                continue
            self.set_status(f"Skipping failed profile with verified X ({attempt}/{attempts})...")
            self.click_once(skip_point)
            self.interruptible_wait(1.2)
            after_capture = self.fresh_capture()
            after_lines = recognize_text(after_capture, Vision)
            similarity = viewport_similarity(
                lines,
                getattr(capture, "frame", None),
                after_lines,
                getattr(after_capture, "frame", None),
            )
            if similarity < 0.90:
                self.interruptible_wait(0.45)
                return True
        return False

    def run_prompt_reply(
        self,
        tone,
        engine="OpenAI API",
        fallback_pickup_line="",
    ):
        started_at = time.monotonic()
        completed = 0
        fallback_likes = 0
        failures = []
        attempted = 0
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            generator = (
                OllamaReplyGenerator()
                if engine == "Local — Free"
                else ReplyGenerator()
            )
            if hasattr(generator, "ensure_ready"):
                self.set_status("Warming up the local reply model...")
                generator.ensure_ready()
            scanner = self.make_profile_scanner(generator)
            rewind_next = False
            for cycle in range(1, self.total_rotations + 1):
                if not self.is_running:
                    break
                attempted += 1
                try:
                    with native_autorelease_pool():
                        reply = self._process_prompt_reply_profile(
                            scanner,
                            generator,
                            tone,
                            cycle,
                            ensure_top=(cycle == 1 or rewind_next),
                        )
                except ScreenRecordingPermissionError:
                    raise
                except Exception as error:
                    skipped = False
                    fallback_sent = False
                    recovery = "uncertain_send"
                    if self.is_running and getattr(
                        self,
                        "_prompt_failure_can_skip",
                        True,
                    ):
                        with native_autorelease_pool():
                            fallback_sent = self.fallback_regular_like(
                                scanner,
                                cycle,
                                fallback_pickup_line,
                                photo_generator=generator,
                                tone=tone,
                            )
                        if fallback_sent:
                            fallback_likes += 1
                            recovery = "pickup_line_sent"
                        else:
                            with native_autorelease_pool():
                                skipped = self.skip_current_profile()
                            recovery = "profile_skipped" if skipped else "recovery_failed"
                    record = self.log_prompt_failure(
                        batch_id,
                        cycle,
                        getattr(self, "_prompt_stage", "unknown"),
                        error,
                        skipped,
                        recovery,
                    )
                    failures.append(record)
                    self.set_status(
                        f"Profile {cycle} failed at {record['stage']}: {error}. "
                        + (
                            "Pickup-line fallback sent; continuing."
                            if fallback_sent
                            else "Skipped; continuing."
                            if skipped
                            else "Could not skip; continuing to the next rotation."
                        )
                    )
                    rewind_next = True
                    self.interruptible_wait(0.3)
                    continue

                completed += 1
                rewind_next = False
                self.set_status(
                    f"Profile {cycle}: sent {reply!r}. Moving to the next profile..."
                )
                self.interruptible_wait(0.25)

            elapsed_seconds = time.monotonic() - started_at
            self.last_prompt_batch = {
                "batch_id": batch_id,
                "attempted": attempted,
                "completed": completed,
                "fallback_likes": fallback_likes,
                "failures": failures,
                "log_path": str(PROMPT_FAILURE_LOG),
                "elapsed_seconds": elapsed_seconds,
            }
            if self.is_running:
                self.set_status(
                    f"Time spent: {format_elapsed_time(elapsed_seconds)}."
                )
            else:
                self.set_status("Stopped.")
        except ScreenRecordingPermissionError as error:
            self.set_status(str(error), show_settings=True)
        except (ProfileScanError, ReplyGenerationError, RuntimeError) as error:
            self.set_status(str(error))
        except Exception as error:
            self.set_status(f"Prompt Reply stopped safely: {error}")
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def on_start(self):
        if self.is_running:
            self.status_label.config(text="Automation is already running.")
            return
        try:
            rotations = int(self.rotations_entry.get().strip())
            if rotations <= 0:
                raise ValueError
        except ValueError:
            self.status_label.config(text="Enter a positive whole number of rotations.")
            return

        workflow = self.workflow_var.get()
        platform = self.platform_var.get()
        engine = self.engine_var.get()
        fallback_pickup_line = self.fallback_line_entry.get().strip()
        if workflow == "Prompt Reply" and platform != "Hinge":
            self.status_label.config(text="Prompt Reply mode currently supports Hinge only.")
            return
        if (
            workflow == "Prompt Reply"
            and engine == "OpenAI API"
            and not os.environ.get("OPENAI_API_KEY")
        ):
            self.status_label.config(
                text="Set OPENAI_API_KEY before using Prompt Reply mode."
            )
            return
        if workflow == "Prompt Reply" and fallback_pickup_line:
            try:
                fallback_pickup_line = validate_fallback_pickup_line(
                    fallback_pickup_line
                )
            except ReplyGenerationError as error:
                self.status_label.config(text=str(error))
                return

        try:
            window = find_iphone_mirroring_window()
            capture_mirroring_window(window)
        except ScreenRecordingPermissionError as error:
            self.render_status(str(error), show_settings=True)
            return
        except RuntimeError as error:
            self.render_status(str(error))
            return

        self.total_rotations = rotations
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        if workflow == "Prompt Reply":
            tone = self.tone_var.get()
            self.render_status(
                f"Starting Hinge Prompt Reply with {tone} tone using {engine}..."
            )
            target = self.run_prompt_reply
            arguments = (tone, engine, fallback_pickup_line)
        else:
            self.render_status(f"Starting automatic {platform} detection...")
            target = self.run_auto_like
            arguments = (platform,)
        thread = threading.Thread(target=target, args=arguments, daemon=True)
        thread.start()

    def stop(self):
        self.is_running = False
        self.status_label.config(text="Stopping...")

    def restart(self):
        self.is_running = False
        python = sys.executable
        os.execl(python, python, *sys.argv)


def main():
    root = tk.Tk()
    ScoopUpApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
