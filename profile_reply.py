"""Local OCR parsing and full-profile scanning for Hinge prompt replies."""

from dataclasses import dataclass
from difflib import SequenceMatcher
from contextlib import nullcontext
import hashlib
import math
import re
import time

try:
    import objc
except ImportError:
    objc = None


MIN_OCR_CONFIDENCE = 0.55
MAX_VIEWPORTS = 14
MAX_TOP_SCROLLS = 30
UI_TEXT = {
    "age",
    "age v",
    "dating intent",
    "dating intent v",
    "height",
    "height v",
    "signals",
    "signals v",
    "hinge",
    "send like",
    "send priority like",
    "add a comment",
    "write a comment",
    "cancel",
    "done",
    "like",
    "likes you",
    "match",
    "new here",
    "agnostic",
    "atheist",
    "buddhist",
    "catholic",
    "christian",
    "hindu",
    "jewish",
    "liberal",
    "moderate",
    "conservative",
    "monogamy",
    "non monogamy",
    "life partner",
    "long term relationship",
    "short term relationship",
    "straight",
    "gay",
    "lesbian",
    "bisexual",
    "pansexual",
    "asexual",
    "white caucasian",
    "east asian",
    "south asian",
    "southeast asian",
    "black african descent",
    "hispanic latino",
    "middle eastern",
    "native american",
    "pacific islander",
    "mixed race",
    "multiracial",
    "non binary",
}
PROMPT_START_WORDS = {
    "a", "all", "best", "change", "dating", "do", "don", "first",
    "give", "green", "guess", "how", "i", "let", "most", "my",
    "never", "one", "something", "teach", "the", "this", "together",
    "try", "two", "typical", "unusual", "we", "what", "when", "you",
}
IGNORED_PROMPT_HEADINGS = {
    "let s get together",
    "lets get together",
}


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
            frame_top = (1 - bounds.origin.y - bounds.size.height) * capture.frame.height
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
    normalized = [normalize_text(line.text) for line in lines if line.confidence >= MIN_OCR_CONFIDENCE]
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


def _is_ui_text(text):
    normalized = normalize_text(text)
    return (
        not normalized
        or normalized in UI_TEXT
        or normalized.startswith("send like")
        or re.match(r"^\d{1,2}\s+\d{2}\b", normalized) is not None
        or re.search(r"\bshows thoughtful signals?\b", normalized) is not None
    )


def _is_ignored_prompt_heading(text):
    normalized = normalize_text(text)
    return (
        normalized in IGNORED_PROMPT_HEADINGS
        or normalized.startswith("let s get together")
        or normalized.startswith("lets get together")
    )


def _looks_like_prompt_heading(text):
    words = normalize_text(text).split()
    return len(words) >= 2 and words[0] in PROMPT_START_WORDS


def prompts_from_viewport(
    lines,
    hearts,
    viewport_index,
    window_height,
    window_top=0,
):
    """Associate text above each Hinge heart with a prompt card."""
    prompts = []
    for heart_point, heart_confidence in hearts:
        heart_x, heart_y = heart_point
        relative_heart_y = (heart_y - window_top) / window_height
        # A clipped heart at the iPhone navigation edge often belongs to the
        # next card and can incorrectly pull profile metadata into its text
        # window. It is not safe to click or associate until scrolled inward.
        if not 0.12 <= relative_heart_y <= 0.88:
            continue
        nearby = [
            line for line in lines
            if line.confidence >= MIN_OCR_CONFIDENCE
            and not _is_ui_text(line.text)
            and line.center[0] < heart_x + 20
            and heart_y - window_height * 0.55 <= line.center[1] <= heart_y + window_height * 0.03
        ]
        nearby.sort(key=lambda line: (line.top, line.left))
        if len(nearby) < 2:
            continue

        # A Hinge prompt label is small text immediately followed by a much
        # larger answer. This font transition separates it from the sticky
        # profile name, status bar, and metadata above the card.
        heading_index = None
        for index in range(len(nearby) - 1):
            heading = nearby[index]
            first_answer = nearby[index + 1]
            vertical_gap = first_answer.top - (heading.top + heading.height)
            if (
                heading.height <= window_height * 0.032
                # Wrapped answer fragments can also be followed by a taller
                # OCR line. They are not prompt headings and must not replace
                # the real title selected above them.
                and not heading.text[:1].islower()
                # Long written prompt headings legitimately extend toward the
                # right-side heart. Anchor association by their left edge so
                # wording length does not cause valid cards to be discarded.
                and heading.left <= heart_x - window_height * 0.08
                and len(normalize_text(heading.text).split()) >= 2
                and first_answer.height >= heading.height * 1.20
                and -3 <= vertical_gap <= window_height * 0.045
            ):
                heading_index = index
        if heading_index is None:
            continue
        nearby = nearby[heading_index:]
        prompt_text = nearby[0].text.strip()
        if _is_ignored_prompt_heading(prompt_text):
            continue
        if not _looks_like_prompt_heading(prompt_text):
            continue
        answer_text = " ".join(line.text.strip() for line in nearby[1:]).strip()
        normalized = normalize_text(f"{prompt_text}|{answer_text}")
        if len(normalized) < 8 or not answer_text:
            continue
        prompt_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        confidence = min(
            heart_confidence,
            sum(line.confidence for line in nearby) / len(nearby),
        )
        prompts.append(CapturedPrompt(
            prompt_id=prompt_id,
            prompt=prompt_text,
            answer=answer_text,
            viewport_index=viewport_index,
            scroll_steps=viewport_index,
            heart_point=heart_point,
            confidence=confidence,
        ))
    return prompts


def merge_prompts(existing, incoming):
    """Merge exact and highly similar prompt cards from overlapping viewports."""
    merged = list(existing)
    for candidate in incoming:
        candidate_text = normalize_text(candidate.combined_text)
        duplicate_index = None
        for index, current in enumerate(merged):
            similarity = SequenceMatcher(
                None,
                candidate_text,
                normalize_text(current.combined_text),
            ).ratio()
            if candidate.prompt_id == current.prompt_id or similarity >= 0.90:
                duplicate_index = index
                break
        if duplicate_index is None:
            merged.append(candidate)
        elif candidate.confidence > merged[duplicate_index].confidence:
            merged[duplicate_index] = candidate
    return merged


def prompts_match(candidate, target):
    candidate_heading = normalize_text(candidate.prompt)
    target_heading = normalize_text(target.prompt)
    if candidate_heading and candidate_heading == target_heading:
        return True
    heading_similarity = SequenceMatcher(
        None,
        candidate_heading,
        target_heading,
    ).ratio()
    candidate_words = set(normalize_text(candidate.answer).split())
    target_words = set(normalize_text(target.answer).split())
    answer_overlap = len(candidate_words & target_words) / max(1, len(target_words))
    return heading_similarity >= 0.88 and answer_overlap >= 0.40


def _prompt_similarity(candidate, target):
    return SequenceMatcher(
        None,
        normalize_text(candidate.combined_text),
        normalize_text(target.combined_text),
    ).ratio()


def _matching_prompts(candidates, target):
    matches = []
    for candidate in candidates:
        similarity = _prompt_similarity(candidate, target)
        if (
            candidate.prompt_id == target.prompt_id
            or similarity >= 0.86
            or prompts_match(candidate, target)
        ):
            matches.append((similarity, candidate))
    matches.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
    return matches


def _unique_prompt_match(candidates, target):
    """Return the best unique match for a selected prompt, or None."""
    matches = _matching_prompts(candidates, target)
    if not matches:
        return None
    if len(matches) > 1 and abs(matches[0][0] - matches[1][0]) < 0.03:
        raise ProfileScanError("The selected prompt has an ambiguous heart target.")
    return matches[0][1]


def prompt_text_vertical_center(lines, target, window_height):
    """Estimate the selected prompt card's text center from fresh OCR."""
    target_heading = normalize_text(target.prompt)
    target_answer_words = set(normalize_text(target.answer).split())
    headings = []
    for line in lines:
        observed = normalize_text(line.text)
        similarity = SequenceMatcher(None, observed, target_heading).ratio()
        if line.confidence >= MIN_OCR_CONFIDENCE and (
            observed == target_heading or similarity >= 0.88
        ):
            headings.append((line.confidence, line))
    if not headings:
        return None
    heading = max(headings, key=lambda item: item[0])[1]
    bottom = heading.top + heading.height
    for line in lines:
        if not (
            line.confidence >= MIN_OCR_CONFIDENCE
            and heading.top <= line.top <= heading.top + window_height * 0.36
        ):
            continue
        words = set(normalize_text(line.text).split())
        if words & target_answer_words:
            bottom = max(bottom, line.top + line.height)
    return (heading.top + bottom) / 2


def recover_visible_prompt_target(target, lines, hearts, viewport_index, window):
    """Recover a prompt heart when OCR grouping breaks after a small nudge."""
    text_center = prompt_text_vertical_center(lines, target, window.height)
    if text_center is None:
        return None
    candidates = []
    for heart_point, confidence in hearts:
        heart_x, heart_y = heart_point
        relative_y = (heart_y - getattr(window, "top", 0)) / window.height
        distance = heart_y - text_center
        if (
            0.18 <= relative_y <= 0.82
            and heart_x >= getattr(window, "left", 0) + window.width * 0.55
            and 0 <= distance <= window.height * 0.32
        ):
            candidates.append((distance, -confidence, heart_point, confidence))
    if not candidates:
        return None
    candidates.sort()
    _, _, heart_point, confidence = candidates[0]
    return CapturedPrompt(
        prompt_id=target.prompt_id,
        prompt=target.prompt,
        answer=target.answer,
        viewport_index=viewport_index,
        scroll_steps=target.scroll_steps,
        heart_point=heart_point,
        confidence=confidence,
    )


def recover_vision_prompt_target(
    prompt_text,
    answer_text,
    lines,
    hearts,
    viewport_index,
    window,
):
    """Anchor Qwen-extracted text to a deterministic OCR line and detected heart."""
    prompt_text = str(prompt_text).strip()
    answer_text = str(answer_text).strip()
    normalized = normalize_text(f"{prompt_text}|{answer_text}")
    if (
        len(normalized) < 8
        or not prompt_text
        or not answer_text
        or _is_ignored_prompt_heading(prompt_text)
        or not _looks_like_prompt_heading(prompt_text)
    ):
        return None
    provisional = CapturedPrompt(
        prompt_id=hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12],
        prompt=prompt_text,
        answer=answer_text,
        viewport_index=viewport_index,
        scroll_steps=viewport_index,
        heart_point=None,
        confidence=0.0,
    )
    return recover_visible_prompt_target(
        provisional,
        lines,
        hearts,
        viewport_index,
        window,
    )


def find_text_target(lines, phrases, min_confidence=MIN_OCR_CONFIDENCE):
    normalized_phrases = tuple(normalize_text(phrase) for phrase in phrases)
    matches = []
    for line in lines:
        normalized = normalize_text(line.text)
        if line.confidence < min_confidence:
            continue
        if any(phrase in normalized for phrase in normalized_phrases):
            matches.append((line.confidence, line.center, line.text))
    return max(matches, default=(0.0, None, ""), key=lambda item: item[0])


def reply_is_visible(lines, reply):
    target = normalize_text(reply)
    if not target:
        return False
    observed_lines = [
        normalize_text(line.text)
        for line in lines
        if line.confidence >= MIN_OCR_CONFIDENCE and normalize_text(line.text)
    ]
    observed = " ".join(observed_lines)
    target_sequence = target.split()
    observed_words = set(observed.split())
    # A dropped space is a real entry failure, even when fuzzy similarity is
    # otherwise high. Reject any OCR token that joins adjacent target words so
    # the composer is cleared and the whole reply is entered again.
    if any(
        first + second in observed_words
        for first, second in zip(target_sequence, target_sequence[1:])
    ):
        return False
    # Hinge's composer is narrow, so Vision frequently returns only the first
    # visible line of a correctly pasted reply. A distinctive 10-character
    # prefix is sufficient evidence while still avoiding a placeholder match.
    if (
        target in observed
        or (len(target) >= 10 and target[:10] in observed)
        or (len(target) >= 10 and target[-10:] in observed)
    ):
        return True
    target_words = {word for word in target.split() if len(word) >= 4}
    matching_words = len(target_words & observed_words)
    required_words = max(2, math.ceil(len(target_words) * 0.45))
    if matching_words >= required_words:
        return True

    # OCR can split or slightly corrupt a word at the edge of the text field.
    # Compare small groups of adjacent OCR lines rather than the whole screen.
    for start in range(len(observed_lines)):
        for width in range(1, min(4, len(observed_lines) - start) + 1):
            fragment = " ".join(observed_lines[start:start + width])
            if len(fragment) >= 12 and SequenceMatcher(None, target, fragment).ratio() >= 0.58:
                return True
    return False


def reply_is_visible_near(lines, reply, center, window_height):
    """Verify reply OCR only inside Hinge's composer, not the prompt above it."""
    if center is None:
        return False
    vertical_radius = window_height * 0.09
    composer_lines = [
        line
        for line in lines
        if abs(line.center[1] - center[1]) <= vertical_radius
    ]
    return reply_is_visible(composer_lines, reply)


def prompt_is_visible(lines, prompt):
    """Confirm that a freshly opened Hinge dialog contains the chosen prompt."""
    observed = " ".join(
        normalize_text(line.text)
        for line in lines
        if line.confidence >= MIN_OCR_CONFIDENCE
    )
    heading = normalize_text(prompt.prompt)
    answer = normalize_text(prompt.answer)
    if not heading or not answer or heading not in observed:
        return False
    if answer in observed:
        return True
    answer_words = {word for word in answer.split() if len(word) >= 4}
    observed_words = set(observed.split())
    required = max(2, math.ceil(len(answer_words) * 0.45))
    return len(answer_words & observed_words) >= required


class ProfileScanner:
    """Scroll a mirrored Hinge profile and collect prompt cards locally."""

    def __init__(
        self,
        capture,
        ocr,
        find_hearts,
        scroll,
        wait,
        is_running=lambda: True,
        progress=lambda _message: None,
        max_viewports=MAX_VIEWPORTS,
        prompt_limit=1,
        vision_rescue=None,
        fast_jump_viewports=0,
        vision_rescue_delay=5.0,
        clock=time.monotonic,
    ):
        self.capture = capture
        self.ocr = ocr
        self.find_hearts = find_hearts
        self.scroll = scroll
        self.wait = wait
        self.is_running = is_running
        self.progress = progress
        self.max_viewports = max_viewports
        self.prompt_limit = max(1, int(prompt_limit))
        self.vision_rescue = vision_rescue
        self.fast_jump_viewports = max(0, int(fast_jump_viewports))
        self.vision_rescue_delay = max(0.0, float(vision_rescue_delay))
        self.clock = clock

    def _current(self):
        capture = self.capture()
        lines = self.ocr(capture)
        return capture, lines

    def _parse_viewport(self, viewport_index):
        capture, lines = self._current()
        hearts = self.find_hearts(capture)
        candidates = prompts_from_viewport(
            lines,
            hearts,
            viewport_index,
            capture.window.height,
            getattr(capture.window, "top", 0),
        )
        return capture, lines, hearts, candidates

    def _rescue_viewport(self, capture, lines, hearts, viewport_index):
        rescued = self.vision_rescue(capture, lines, hearts, viewport_index)
        return [rescued] if rescued is not None else []

    def scroll_to_top(self):
        stable = 0
        previous_capture = None
        previous_lines = None
        for attempt in range(MAX_TOP_SCROLLS):
            if not self.is_running():
                raise ProfileScanError("Profile scan stopped.")
            self.progress(f"Returning to the top ({attempt + 1}/{MAX_TOP_SCROLLS})...")
            capture, lines = self._current()
            similarity = (
                viewport_similarity(
                    previous_lines,
                    getattr(previous_capture, "frame", None),
                    lines,
                    getattr(capture, "frame", None),
                )
                if previous_lines is not None else 0.0
            )
            if similarity >= 0.94:
                stable += 1
                if stable >= 2:
                    return lines
            else:
                stable = 0
            previous_capture = capture
            previous_lines = lines
            self.scroll("up")
            self.wait(0.5)
        raise ProfileScanError("Could not reliably reach the top of the profile.")

    def _same_viewport(self, previous_capture, previous_lines, capture, lines):
        return (
            previous_lines is not None
            and viewport_similarity(
                previous_lines,
                getattr(previous_capture, "frame", None),
                lines,
                getattr(capture, "frame", None),
            ) >= 0.94
        )

    def scan(self, ensure_top=True):
        if ensure_top:
            self.scroll_to_top()
        if self.fast_jump_viewports:
            self.progress("Jumping directly to the likely first-prompt area...")
            for _ in range(self.fast_jump_viewports):
                self.scroll("down")
                self.wait(0.22)
        scan_started = self.clock()
        prompts = []
        viewport_count = 0
        previous_capture = None
        previous_lines = None
        no_new_content = 0
        vision_rescue_used = False

        for viewport_index in range(self.max_viewports):
            if not self.is_running():
                raise ProfileScanError("Profile scan stopped.")
            self.progress(
                f"Scanning profile viewport {viewport_index + 1}/{self.max_viewports}..."
            )
            capture, lines, hearts, candidates = self._parse_viewport(viewport_index)
            if (
                not candidates
                and self.vision_rescue is not None
                and not vision_rescue_used
                and self.clock() - scan_started >= self.vision_rescue_delay
            ):
                vision_rescue_used = True
                self.progress(
                    f"Using local vision to rescue viewport {viewport_index + 1}..."
                )
                candidates = self._rescue_viewport(
                    capture, lines, hearts, viewport_index
                )
            before = len(prompts)
            prompts = merge_prompts(prompts, candidates)
            viewport_count += 1
            if len(prompts) >= self.prompt_limit:
                self.progress(
                    "Found the first written prompt; starting the reply workflow..."
                )
                break
            if len(prompts) == before and self._same_viewport(
                previous_capture, previous_lines, capture, lines
            ):
                no_new_content += 1
            else:
                no_new_content = 0
            previous_capture = capture
            previous_lines = lines
            if no_new_content >= 2:
                break
            if viewport_index < self.max_viewports - 1:
                self.scroll("down")
                self.wait(0.4)

        if not prompts and self.vision_rescue is not None and not vision_rescue_used:
            remaining = self.vision_rescue_delay - (self.clock() - scan_started)
            if remaining > 0:
                self.progress(
                    f"No OCR prompt yet; waiting {remaining:.1f}s before local vision rescue..."
                )
                self.wait(remaining)
            if not self.is_running():
                raise ProfileScanError("Profile scan stopped.")
            capture, lines, hearts, candidates = self._parse_viewport(viewport_count)
            if not candidates:
                self.progress(
                    "No prompt found by OCR after "
                    f"{self.vision_rescue_delay:g} seconds; using local vision..."
                )
                candidates = self._rescue_viewport(
                    capture, lines, hearts, viewport_count
                )
            prompts = merge_prompts(prompts, candidates)
            viewport_count += 1

        if not prompts:
            raise ProfileScanError("No readable Hinge prompt and answer pairs were found.")
        fingerprint_payload = "|".join(sorted(prompt.prompt_id for prompt in prompts))
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:20]
        return ProfileScan(tuple(prompts), viewport_count, fingerprint)

    def relocate(self, target):
        """Re-find a selected prompt from fresh captures before allowing a click."""
        self.scroll_to_top()
        previous_capture = None
        previous_lines = None
        unchanged_views = 0
        for viewport_index in range(self.max_viewports):
            capture, lines, _hearts, candidates = self._parse_viewport(viewport_index)
            match = _unique_prompt_match(candidates, target)
            if match is not None:
                return match
            unchanged_views = (
                unchanged_views + 1
                if self._same_viewport(previous_capture, previous_lines, capture, lines)
                else 0
            )
            if unchanged_views >= 2:
                break
            previous_capture = capture
            previous_lines = lines
            if viewport_index < self.max_viewports - 1:
                self.scroll("down")
                self.wait(0.4)
        raise ProfileScanError("The selected prompt changed before it could be opened.")

    def reconfirm_visible(self, target, attempts=3):
        """Re-confirm the selected prompt in place without rewinding the profile."""
        for attempt in range(attempts):
            if not self.is_running():
                raise ProfileScanError("Profile scan stopped.")
            _capture, _lines, _hearts, candidates = self._parse_viewport(
                target.viewport_index
            )
            match = _unique_prompt_match(candidates, target)
            if match is not None:
                return match
            if attempt < attempts - 1:
                self.wait(0.25)
        raise ProfileScanError(
            "The first written prompt was no longer visible after generation; nothing was clicked."
        )

    def center_target(self, target, initial, attempts=14):
        """Center the chosen prompt card while keeping its heart safely tappable."""
        current = initial
        last_direction = "down_small"
        last_relative_y = None
        capture, _lines = self._current()
        window = capture.window
        for _ in range(attempts):
            relative_y = (
                (current.heart_point[1] - window.top) / window.height
                if current is not None
                else None
            )
            last_relative_y = relative_y
            if (
                current is not None
                and relative_y is not None
                and relative_y <= 0.70
            ):
                return current

            # Hearts in the top 70% are already safely tappable. Only nudge
            # upward when the target sits in the bottom 30% near navigation.
            direction = last_direction if relative_y is None else "down_small"
            last_direction = direction
            self.scroll(direction)
            self.wait(0.18)
            capture, lines, hearts, candidates = self._parse_viewport(
                target.viewport_index
            )
            window = capture.window
            matches = _matching_prompts(candidates, target)
            if not matches:
                observed = " ".join(normalize_text(line.text) for line in lines)
                if normalize_text(target.prompt) in observed:
                    current = recover_visible_prompt_target(
                        target,
                        lines,
                        hearts,
                        target.viewport_index,
                        capture.window,
                    )
                    continue
                # Vision can miss a heading for one frame immediately after a
                # scroll. Re-read in place before declaring the prompt lost.
                try:
                    current = self.reconfirm_visible(target, attempts=2)
                    capture, lines, hearts, _candidates = self._parse_viewport(
                        target.viewport_index
                    )
                    window = capture.window
                    continue
                except ProfileScanError:
                    raise ProfileScanError(
                        "The selected prompt was lost while moving it to the screen center."
                    )
            current = matches[0][1]
        raise ProfileScanError(
            "The selected prompt could not be centered safely away from iPhone navigation "
            f"(last heart position: {last_relative_y!r})."
        )
