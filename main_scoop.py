import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

from pynput.mouse import Button, Controller

try:
    import AppKit
    import Quartz
    import Vision
except ImportError:
    AppKit = None
    Quartz = None
    Vision = None


IPHONE_MIRRORING_BUNDLE_ID = "com.apple.ScreenContinuity"
SCREEN_RECORDING_SETTINGS_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
)

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


def capture_mirroring_window(window=None):
    """Capture only the latest visible iPhone Mirroring window."""
    window = window or find_iphone_mirroring_window()
    image = Quartz.CGWindowListCreateImage(
        # CGRectNull asks Quartz to derive the exact window-image bounds. Using
        # desktop bounds here can crop or horizontally offset Retina windows.
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window.window_id,
        Quartz.kCGWindowImageBoundsIgnoreFraming,
    )
    if image is None:
        raise ScreenRecordingPermissionError(
            "Could not capture iPhone Mirroring. Allow Screen Recording access in System Settings."
        )

    provider = Quartz.CGImageGetDataProvider(image)
    pixels = bytes(Quartz.CGDataProviderCopyData(provider))
    frame = PixelFrame(
        Quartz.CGImageGetWidth(image),
        Quartz.CGImageGetHeight(image),
        Quartz.CGImageGetBytesPerRow(image),
        pixels,
    )
    return WindowCapture(window, image, frame)


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

    def find(self, capture, platform):
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
            return None, 0.0
        candidates.sort(reverse=True)
        score, center_x, center_y, _ = candidates[0]
        return capture.desktop_point(center_x, center_y), min(score, 1.0)


def _normalize_text(text):
    return re.sub(r"[^a-z]+", " ", text.casefold()).strip()


def find_send_priority_like(capture):
    """Use native macOS OCR to find Hinge's current confirmation button."""
    if Vision is None:
        raise RuntimeError(
            "Text detection requires pyobjc-framework-Vision. Install dependencies first."
        )

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        capture.image,
        {},
    )
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"Text detection failed: {error}")

    candidates = []
    for observation in request.results() or []:
        recognized = observation.topCandidates_(1)
        if not recognized:
            continue
        text = str(recognized[0].string())
        normalized = _normalize_text(text)
        words = set(normalized.split())
        if not {"send", "like"}.issubset(words):
            continue

        bounds = observation.boundingBox()
        frame_x = (bounds.origin.x + bounds.size.width / 2) * capture.frame.width
        frame_y = (
            1 - (bounds.origin.y + bounds.size.height / 2)
        ) * capture.frame.height
        score = float(recognized[0].confidence())
        if "priority" in words:
            score += 0.25
        candidates.append((score, capture.desktop_point(frame_x, frame_y), text))

    if not candidates:
        return None, 0.0, ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, point, text = candidates[0]
    return point, min(score, 1.0), text


class ScoopUpApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Scoop UP — Automatic Detection")
        self.root.geometry("430x340")
        self.root.resizable(True, False)

        self.mouse = Controller()
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

    def find_send_like_before(self, deadline):
        """Run native OCR without allowing it to hold up the cycle past its deadline."""
        results = queue.Queue(maxsize=1)

        def detect():
            try:
                capture = self.fresh_capture()
                results.put(("result", find_send_priority_like(capture)))
            except Exception as error:
                results.put(("error", error))

        threading.Thread(target=detect, daemon=True).start()
        remaining = max(0, deadline - time.monotonic())
        try:
            kind, value = results.get(timeout=remaining)
        except queue.Empty:
            return None, 0.0, ""
        if kind == "error":
            raise value
        return value

    def find_heart_before(self, platform, deadline):
        """Detect the current heart without holding a cycle past its deadline."""
        results = queue.Queue(maxsize=1)

        def detect():
            try:
                capture = self.fresh_capture()
                results.put(("result", self.heart_detector.find(capture, platform)))
            except Exception as error:
                results.put(("error", error))

        threading.Thread(target=detect, daemon=True).start()
        remaining = max(0, deadline - time.monotonic())
        try:
            kind, value = results.get(timeout=remaining)
        except queue.Empty:
            return None, 0.0, True
        if kind == "error":
            raise value
        point, score = value
        return point, score, False

    def run(self, platform):
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
        platform = self.platform_var.get()
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.render_status(f"Starting automatic {platform} detection...")
        thread = threading.Thread(target=self.run, args=(platform,), daemon=True)
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
