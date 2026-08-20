"""macOS iPhone Mirroring window discovery and screen capture."""

from contextlib import nullcontext
from dataclasses import dataclass
import threading

try:
    import AppKit
    import objc
    import Quartz
except ImportError:
    AppKit = None
    objc = None
    Quartz = None

try:
    import ScreenCaptureKit
except ImportError:
    ScreenCaptureKit = None


IPHONE_MIRRORING_BUNDLE_ID = "com.apple.ScreenContinuity"


def native_autorelease_pool():
    return objc.autorelease_pool() if objc is not None else nullcontext()


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
