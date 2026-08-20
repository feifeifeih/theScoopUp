"""Vision OCR primitives and viewport comparison for profile scanning."""

from contextlib import nullcontext
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import re

try:
    import objc
except ImportError:
    objc = None


MIN_OCR_CONFIDENCE = 0.55


def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


@dataclass(frozen=True)
class ScreenText:
    text: str
    confidence: float
    left: float
    top: float
    width: float
    height: float

    @property
    def center(self):
        return self.left + self.width / 2, self.top + self.height / 2


@dataclass(frozen=True)
class CapturedPrompt:
    prompt_id: str
    prompt: str
    answer: str
    viewport_index: int
    scroll_steps: int
    heart_point: tuple
    confidence: float

    @property
    def combined_text(self):
        return f"{self.prompt} {self.answer}".strip()


@dataclass(frozen=True)
class ProfileScan:
    prompts: tuple
    viewport_count: int
    profile_fingerprint: str


class ProfileScanError(RuntimeError):
    pass


def recognize_text(capture, vision_module, accurate=True):
    """Recognize visible text and return desktop-coordinate bounding boxes."""
    if vision_module is None:
        raise RuntimeError(
            "Text detection requires pyobjc-framework-Vision. Install dependencies first."
        )

    lines = []
    pool = objc.autorelease_pool() if objc is not None else nullcontext()
    with pool:
        request = vision_module.VNRecognizeTextRequest.alloc().init()
        if accurate:
            request.setRecognitionLevel_(
                vision_module.VNRequestTextRecognitionLevelAccurate
            )
            request.setUsesLanguageCorrection_(True)
        else:
            request.setRecognitionLevel_(
                getattr(
                    vision_module,
                    "VNRequestTextRecognitionLevelFast",
                    vision_module.VNRequestTextRecognitionLevelAccurate,
                )
            )
            request.setUsesLanguageCorrection_(False)
        image = (
            capture.ensure_vision_image()
            if hasattr(capture, "ensure_vision_image")
            else capture.image
        )
        handler = vision_module.VNImageRequestHandler.alloc().initWithCGImage_options_(
            image,
            {},
        )
        success, error = handler.performRequests_error_([request], None)
        if not success:
            raise RuntimeError(f"Text detection failed: {error}")

        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            bounds = observation.boundingBox()
            frame_left = bounds.origin.x * capture.frame.width
            frame_top = (
                1 - bounds.origin.y - bounds.size.height
            ) * capture.frame.height
            desktop_left, desktop_top = capture.desktop_point(frame_left, frame_top)
            lines.append(ScreenText(
                text=str(candidate.string()).strip(),
                confidence=float(candidate.confidence()),
                left=desktop_left,
                top=desktop_top,
                width=bounds.size.width * capture.window.width,
                height=bounds.size.height * capture.window.height,
            ))
    return sorted(lines, key=lambda line: (line.top, line.left))


def _frame_color_samples(frame):
    """Sample a coarse color grid so photo-only viewports still have a signature."""
    for row in range(1, 12):
        for column in range(1, 8):
            yield frame.color_at(frame.width * column / 8, frame.height * row / 12)


def viewport_signature(lines, frame=None):
    normalized = [
        normalize_text(line.text)
        for line in lines
        if line.confidence >= MIN_OCR_CONFIDENCE
    ]
    parts = [part for part in normalized if part]
    if frame is not None:
        for color in _frame_color_samples(frame):
            parts.append("".join(f"{channel // 32:x}" for channel in color))
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def viewport_similarity(first_lines, first_frame, second_lines, second_frame):
    """Compare OCR and sampled pixels while tolerating small recognition jitter."""
    first_text = "|".join(normalize_text(line.text) for line in first_lines)
    second_text = "|".join(normalize_text(line.text) for line in second_lines)
    text_score = SequenceMatcher(None, first_text, second_text).ratio()

    if first_frame is None or second_frame is None:
        return text_score
    color_score = 0.0
    sample_count = 0
    for first_color, second_color in zip(
        _frame_color_samples(first_frame),
        _frame_color_samples(second_frame),
    ):
        channel_difference = sum(
            abs(first - second) for first, second in zip(first_color, second_color)
        ) / (len(first_color) * 255)
        color_score += 1 - channel_difference
        sample_count += 1
    return 0.55 * text_score + 0.45 * (color_score / max(1, sample_count))
