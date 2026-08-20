"""Prompt parsing and full-profile scanning for Hinge prompt replies."""

import hashlib
import time

from profile_prompts import (
    IGNORED_PROMPT_HEADINGS,
    PROMPT_START_WORDS,
    UI_TEXT,
    _matching_prompts,
    _unique_prompt_match,
    find_text_target,
    merge_prompts,
    prompt_is_visible,
    prompt_text_vertical_center,
    prompts_from_viewport,
    prompts_match,
    recover_visible_prompt_target,
    recover_vision_prompt_target,
    reply_is_visible,
    reply_is_visible_near,
)
from profile_vision import (
    MIN_OCR_CONFIDENCE,
    CapturedPrompt,
    ProfileScan,
    ProfileScanError,
    ScreenText,
    normalize_text,
    recognize_text,
    viewport_signature,
    viewport_similarity,
)


MAX_VIEWPORTS = 14
MAX_TOP_SCROLLS = 30


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
