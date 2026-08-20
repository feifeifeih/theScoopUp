import math
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button, Controller

from heart_detection import HeartIconDetector, HINGE_HEART_BUTTON_TEMPLATE
from mac_capture import (
    IPHONE_MIRRORING_BUNDLE_ID,
    MirroringWindow,
    PixelFrame,
    ScreenCaptureBackend,
    ScreenRecordingPermissionError,
    WindowCapture,
    _window_bundle_id,
    _window_score,
    capture_mirroring_window,
    find_iphone_mirroring_window,
)
from prompt_logging import (
    PROMPT_TRANSCRIPT_DIR,
    format_elapsed_time,
    format_prompt_transcript,
    format_prompt_transcript_header,
    parse_openai_api_key,
    prompt_transcript_path,
)
from screen_images import (
    _encode_frame_region_png,
    _png_chunk,
    first_profile_photo_png,
    prompt_viewport_png,
)

from profile_reply import (
    ProfileScanError,
    ProfileScanner,
    find_text_target,
    normalize_text,
    prompt_is_visible,
    recognize_text,
    recover_vision_prompt_target,
    reply_is_visible,
    reply_is_visible_near,
    viewport_similarity,
)
from reply_generation import (
    API_KEY_ENV_NAMES,
    CloudReplyGenerator,
    OllamaReplyGenerator,
    PAID_ENGINE,
    PAY_MODEL,
    PAY_MODELS,
    ReplyGenerationError,
    ReplyGenerator,
    TONE_INSTRUCTIONS,
    build_input,
    paid_model_from_selection,
    random_pickup_line,
)

PAID_MODEL_PROMPT = "Select a model…"


def is_paid_engine(engine):
    return engine in {PAID_ENGINE, "OpenAI API"}


def is_no_prompt_scan_failure(stage, error):
    return stage == "scan" and "No readable Hinge prompt" in str(error)


def paid_photo_fallback_enabled(engine, stage, error):
    if engine == "Local — Free":
        return True
    if is_paid_engine(engine):
        return is_no_prompt_scan_failure(stage, error)
    return False


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

SCREEN_RECORDING_SETTINGS_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
)
PROMPT_FAILURE_LOG = (
    Path.home() / "Library" / "Logs" / "The Scoop UP" / "prompt_reply_failures.jsonl"
)


def native_autorelease_pool():
    """Drain temporary PyObjC objects at the end of one bounded operation."""
    return objc.autorelease_pool() if objc is not None else nullcontext()


SEND_SETTLE_TIMEOUT = 1.1
SEND_POLL_INTERVAL = 0.15
SEND_LIKE_MIN_SCORE = 0.55
COMPOSER_VERTICAL_RATIO = 0.085
TYPE_CHARACTER_DELAY = 0.018
TYPE_SPACE_DELAY = 0.045


def comment_point_from_send(window, send_point):
    return (
        window.left + window.width * 0.50,
        send_point[1] - window.height * COMPOSER_VERTICAL_RATIO,
    )


def send_point_from_comment(window, comment_point):
    return (
        window.left + window.width * 0.62,
        comment_point[1] + window.height * COMPOSER_VERTICAL_RATIO,
    )


def is_safe_iphone_action_point(capture, point, *, bottom_limit=0.86):
    """Reject action coordinates near iPhone's status and home gesture areas."""
    if point is None:
        return False
    window = capture.window
    relative_x = (point[0] - window.left) / window.width
    relative_y = (point[1] - window.top) / window.height
    return 0.04 <= relative_x <= 0.96 and 0.12 <= relative_y <= bottom_limit


def find_send_priority_like(capture, lines=None):
    """Use native macOS OCR to find Hinge's current confirmation button."""
    if lines is None:
        lines = recognize_text(capture, Vision)
    candidates = []
    for line in lines:
        normalized = normalize_text(line.text)
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
        self.root.title("The Scoop UP")
        self.root.geometry("380x320")
        self.root.resizable(False, False)

        self.mouse = Controller()
        self.keyboard = KeyboardController()
        self.clipboard = MacClipboard(AppKit)
        self.heart_detector = HeartIconDetector()
        self.is_running = False
        self.total_rotations = 20
        self._prompt_stage = "unknown"
        self._prompt_failure_can_skip = True
        self.last_prompt_batch = None
        self._save_transcripts = False
        self._prompt_transcript = {}
        self._transcript_log_path = None

        tk.Label(
            root,
            text="iPhone Mirroring Detection",
            font=("Helvetica", 13, "bold"),
        ).pack(pady=(8, 4))

        form = tk.Frame(root)
        form.pack(padx=12)
        form.columnconfigure(1, weight=1)
        self.form = form

        self.platform_var = tk.StringVar(value="Hinge")
        self.workflow_var = tk.StringVar(value="Auto Like")
        self.engine_var = tk.StringVar(value="Local — Free")
        self.paid_model_var = tk.StringVar(value=PAID_MODEL_PROMPT)
        self._provider_keys = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", ""),
            "XAI_API_KEY": os.environ.get("XAI_API_KEY", ""),
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        }
        self._key_env = "OPENAI_API_KEY"
        self.api_key_var = tk.StringVar(value="")
        self.tone_var = tk.StringVar(value="Playful & clean")
        self.rotations_entry = tk.Entry(form, width=14, justify="center")
        self.rotations_entry.insert(0, "20")
        self.save_transcript_var = tk.BooleanVar(value=False)

        api_key_row = tk.Frame(form)
        self.api_key_entry = tk.Entry(
            api_key_row,
            textvariable=self.api_key_var,
            show="*",
            width=18,
        )
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            api_key_row,
            text="Import",
            command=self.import_api_key,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 0))

        self._base_form_rows = (
            (
                tk.Label(form, text="Dating app:"),
                tk.OptionMenu(form, self.platform_var, "Hinge", "Tinder"),
            ),
            (
                tk.Label(form, text="Workflow:"),
                tk.OptionMenu(form, self.workflow_var, "Auto Like", "Prompt Reply"),
            ),
            (
                tk.Label(form, text="Reply engine:"),
                tk.OptionMenu(form, self.engine_var, "Local — Free", PAID_ENGINE),
            ),
            (
                tk.Label(form, text="Reply tone:"),
                tk.OptionMenu(form, self.tone_var, *TONE_INSTRUCTIONS.keys()),
            ),
            (tk.Label(form, text="Rotations:"), self.rotations_entry),
        )
        self._save_transcript_row = (
            tk.Label(form, text="Save log:"),
            tk.Checkbutton(
                form,
                text="Prompt & reply",
                variable=self.save_transcript_var,
                anchor="w",
            ),
        )
        self.paid_model_menu = tk.OptionMenu(
            form,
            self.paid_model_var,
            PAID_MODEL_PROMPT,
            *PAY_MODELS,
        )
        self._paid_model_row = (
            tk.Label(form, text="Paid model:"),
            self.paid_model_menu,
        )
        self._api_key_row = (tk.Label(form, text="API key:"), api_key_row)

        self.engine_var.trace_add("write", lambda *_args: self._layout_form())
        self.paid_model_var.trace_add("write", lambda *_args: self._layout_form())
        self.workflow_var.trace_add("write", lambda *_args: self._layout_form())
        self._layout_form()

        button_frame = tk.Frame(root)
        button_frame.pack(pady=(8, 4))
        self.start_button = tk.Button(button_frame, text="Start", command=self.on_start, width=8)
        self.start_button.grid(row=0, column=0, padx=3)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop, width=8)
        self.stop_button.grid(row=0, column=1, padx=3)
        self.restart_button = tk.Button(button_frame, text="Restart", command=self.restart, width=8)
        self.restart_button.grid(row=0, column=2, padx=3)

        self.status_label = tk.Label(
            root,
            text="Open iPhone Mirroring, choose an app, then press Start.",
            wraplength=360,
            justify="center",
        )
        self.status_label.pack(padx=10, pady=(2, 8))

        self.settings_button = tk.Button(
            root,
            text="Open Screen Recording Settings",
            command=self.open_screen_recording_settings,
        )
        # This button is intentionally hidden until capture permission fails.
        self.root.bind("<Escape>", lambda _event: self.stop())
        self._fit_window()

    def _selected_paid_model(self):
        return paid_model_from_selection(self.paid_model_var.get().strip())

    def _sync_provider_key(self):
        choice = self._selected_paid_model()
        current = self.api_key_var.get()
        if self._key_env:
            self._provider_keys[self._key_env] = current
        if choice is None:
            self._key_env = "OPENAI_API_KEY"
            return
        self._key_env = choice.env_key
        self.api_key_var.set(self._provider_keys.get(choice.env_key, ""))

    def _layout_form(self):
        for child in self.form.grid_slaves():
            child.grid_forget()
        rows = list(self._base_form_rows)
        paid = is_paid_engine(self.engine_var.get())
        if paid:
            self._sync_provider_key()
            rows[3:3] = [self._paid_model_row]
            if self._selected_paid_model():
                rows[4:4] = [self._api_key_row]
        if self.workflow_var.get() == "Prompt Reply":
            rows.append(self._save_transcript_row)
        for index, (label, widget) in enumerate(rows):
            label.grid(row=index, column=0, sticky="e", padx=(0, 8), pady=2)
            widget.grid(row=index, column=1, sticky="ew", pady=2)
        if hasattr(self, "status_label") and not self.is_running:
            choice = self._selected_paid_model()
            if paid and choice is None:
                message = "Select an available paid model."
            elif paid and not self.api_key_var.get().strip():
                message = f"Import or paste your {choice.provider_title} API key."
            else:
                message = "Open iPhone Mirroring, choose an app, then press Start."
            self.status_label.config(text=message)
            self._fit_window()

    def import_api_key(self):
        choice = self._selected_paid_model()
        provider = choice.provider_title if choice else "paid model"
        env_names = (choice.env_key,) if choice else API_KEY_ENV_NAMES
        path = filedialog.askopenfilename(
            title=f"Import {provider} API key",
            filetypes=(
                ("Text files", "*.txt"),
                ("Env files", "*.env"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as error:
            self.render_status(f"Could not read the API key file: {error}")
            return
        key = parse_openai_api_key(text, env_names=env_names + API_KEY_ENV_NAMES)
        if not key:
            self.render_status(f"The selected file did not contain a {provider} API key.")
            return
        self.api_key_var.set(key)
        if choice is not None:
            self._provider_keys[choice.env_key] = key
        self.render_status(f"{provider} API key imported.")

    def _fit_window(self):
        self.root.update_idletasks()
        width = max(self.root.winfo_reqwidth(), 360)
        height = self.root.winfo_reqheight()
        self.root.geometry(f"{width}x{height}")

    def render_status(self, message, show_settings=False):
        self.status_label.config(text=message)
        if show_settings:
            if not self.settings_button.winfo_manager():
                self.settings_button.pack(pady=(4, 8))
        else:
            self.settings_button.pack_forget()
        self._fit_window()

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

    def paste_reply(self, reply, *, defer_restore=False):
        saved = self.clipboard.snapshot()
        self.clear_reply_field()
        self.clipboard.set_text(reply)
        # Give iPhone Mirroring time to observe the new pasteboard change
        # before requesting paste; otherwise it can reuse the prior value.
        self.interruptible_wait(0.4)
        self.keyboard.press(Key.cmd)
        self.keyboard.press("v")
        self.keyboard.release("v")
        self.keyboard.release(Key.cmd)
        # iPhone Mirroring consumes pasteboard data asynchronously. Keep
        # our temporary clipboard value alive until the phone has read it.
        self.interruptible_wait(1.2)
        if not defer_restore:
            self.clipboard.restore(saved)
        return saved

    def type_reply(self, reply):
        if not reply.isascii():
            raise ProfileScanError(
                "Direct typing fallback requires an ASCII reply."
            )
        self.clear_reply_field()
        for character in reply:
            if not self.is_running:
                return
            if character == " ":
                # Mirroring can lose a space embedded in a fast type() stream.
                # Send it as its own key event and give the phone time to apply it.
                self.keyboard.press(Key.space)
                self.keyboard.release(Key.space)
                time.sleep(TYPE_SPACE_DELAY)
            else:
                self.keyboard.type(character)
                time.sleep(TYPE_CHARACTER_DELAY)

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
            and score >= SEND_LIKE_MIN_SCORE
            and is_safe_iphone_action_point(capture, point)
        ):
            return point, score, label
        # Vision sometimes garbles the pale button label. Once the matching
        # dialog and composer text are independently verified, its control is
        # reliably the wide button immediately below the composer.
        fallback = send_point_from_comment(capture.window, comment_point)
        if is_safe_iphone_action_point(capture, fallback):
            return fallback, 0.60, "Send Like (verified layout)"
        return None, 0.0, ""

    def position_open_dialog_safely(self, attempts=7):
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
            if (
                comment_point is None
                and raw_send is not None
                and raw_score >= SEND_LIKE_MIN_SCORE
            ):
                comment_point = comment_point_from_send(capture.window, raw_send)
            candidate_send = raw_send
            candidate_score = raw_score
            if comment_point is not None and candidate_send is None:
                candidate_send = send_point_from_comment(capture.window, comment_point)
                candidate_score = 0.60
            if (
                comment_point is not None
                and candidate_send is not None
                and candidate_score >= SEND_LIKE_MIN_SCORE
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

    def enter_reply(
        self,
        comment_point,
        reply,
        window_height,
        attempts=3,
    ):
        """Focus, replace, enter, and composer-verify with bounded retries."""
        last_capture = None
        last_lines = []
        saved_clipboard = None
        try:
            for attempt in range(1, attempts + 1):
                self.set_status(f"Entering generated reply ({attempt}/{attempts})...")
                self.click_once(comment_point)
                self.interruptible_wait(0.15)
                self.click_once(comment_point)
                self.interruptible_wait(0.35)
                if attempt == 1:
                    # Paste is atomic, so a focus/cursor event cannot join words
                    # halfway through entry. Slower direct typing remains a
                    # fallback for Mirroring sessions that reject paste.
                    saved_clipboard = self.paste_reply(reply, defer_restore=True)
                elif reply.isascii():
                    self.type_reply(reply)
                else:
                    self.paste_reply(reply, defer_restore=True)
                self.interruptible_wait(0.2)
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
        finally:
            if saved_clipboard is not None:
                self.clipboard.restore(saved_clipboard)
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
            if send_point is not None and send_score >= SEND_LIKE_MIN_SCORE:
                # The send control positively identifies the Hinge like sheet.
                # Its composer occupies the full-width box immediately above
                # that control, even when its pale placeholder is not OCR'd.
                composer_point = comment_point_from_send(capture.window, send_point)
                window = capture.window
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

    def open_prompt_dialog(self, scanner, selected, relocated):
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
        self._prompt_transcript = {}
        self.set_status(f"Profile {cycle}: looking for the first written prompt...")
        scan = scanner.scan(ensure_top=ensure_top)
        selected = scan.prompts[0]
        model_input = getattr(generator, "model_input", build_input)
        self._prompt_transcript = {
            "profile_prompt": selected.prompt,
            "profile_answer": selected.answer,
            "model_input": model_input(scan.prompts, tone),
            "model_reply": None,
        }

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
        dialog_capture, dialog_lines, comment_point = self.position_open_dialog_safely()
        if not prompt_is_visible(dialog_lines, selected):
            _capture, dialog_lines = self.wait_for_matching_prompt_dialog(selected)
            if not prompt_is_visible(dialog_lines, selected):
                raise ProfileScanError(
                    "The opened Hinge dialog did not match the generated prompt; nothing was entered or sent."
                )

        self._prompt_stage = "generate"
        self.set_status(f"Profile {cycle}: Send Like is positioned; finishing the reply...")
        generated = self._finish_reply_generation(generation, selected)
        self._prompt_transcript["model_reply"] = generated.reply
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
            if send_point is None or send_score < SEND_LIKE_MIN_SCORE:
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

    def log_prompt_transcript(
        self,
        outcome,
        batch_id,
        cycle,
        tone,
        engine,
        model,
        error=None,
        stage=None,
    ):
        """Append one profile transcript when the save-log checkbox is on."""
        if not self._save_transcripts:
            return None
        snapshot = getattr(self, "_prompt_transcript", None) or {}
        reply = snapshot.get("model_reply")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch_id,
            "rotation": cycle,
            "tone": tone,
            "engine": engine,
            "model": model,
            "outcome": outcome,
            "profile_prompt": snapshot.get("profile_prompt") or None,
            "profile_answer": snapshot.get("profile_answer") or None,
            "model_input": snapshot.get("model_input"),
            "model_reply": reply,
            "sent_to_profile": reply if outcome == "sent" else None,
        }
        if outcome == "failed":
            record["stage"] = stage
            record["error"] = str(error) if error is not None else None
        try:
            path = getattr(self, "_transcript_log_path", None) or prompt_transcript_path()
            self._transcript_log_path = path
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(format_prompt_transcript(record))
        except OSError:
            pass
        return record

    def _hinge_hearts_in_band(self, capture, top=0.18, bottom=0.72):
        window = capture.window
        return [
            (point, score)
            for point, score in self.heart_detector.find_all(capture, "Hinge")
            if top <= (point[1] - window.top) / window.height <= bottom
            and score >= SEND_LIKE_MIN_SCORE
        ]

    def _fallback_hearts_in_band(self, capture):
        """Photo likes place hearts low on the card; allow a wider band than prompts."""
        window = capture.window
        hearts = []
        for point, score in self.heart_detector.find_all(capture, "Hinge"):
            relative_y = (point[1] - window.top) / window.height
            if (
                0.12 <= relative_y <= 0.94
                and score >= SEND_LIKE_MIN_SCORE
                and is_safe_iphone_action_point(
                    capture,
                    point,
                    bottom_limit=0.94,
                )
            ):
                hearts.append((point, score))
        return hearts

    def _capture_fallback_heart(self, scanner):
        """Scroll until a Hinge heart is in the safe middle band for clicking."""
        scanner.scroll_to_top()
        self.interruptible_wait(0.3)
        capture = self.fresh_capture()
        hearts = self._fallback_hearts_in_band(capture)
        for adjust in range(7):
            if hearts:
                return capture, min(hearts, key=lambda item: item[0][1])
            if adjust < 6:
                self.scroll_profile("up" if adjust % 2 == 0 else "down_small")
                self.interruptible_wait(0.35)
                capture = self.fresh_capture()
                hearts = self._fallback_hearts_in_band(capture)
        return capture, None

    def fallback_regular_like(
        self,
        scanner,
        cycle,
        photo_generator=None,
        tone="Playful & clean",
        use_photo_generator=True,
    ):
        """Recover a failed prompt with a verified clean pickup-line comment."""
        pickup_line = None
        self.set_status(
            f"Profile {cycle}: prompt failed; preparing a photo-grounded fallback..."
        )
        capture = self.fresh_capture()
        send_point, _, _ = find_send_priority_like(capture)
        if send_point is not None:
            # Dismiss any leftover like sheet so hearts and photos are visible
            # again before opening a fresh photo-grounded comment.
            self.keyboard.press(Key.esc)
            self.keyboard.release(Key.esc)
            self.interruptible_wait(0.5)

        capture, heart_match = self._capture_fallback_heart(scanner)
        if heart_match is None:
            return False
        heart_point, _ = heart_match

        if (
            pickup_line is None
            and use_photo_generator
            and hasattr(photo_generator, "generate_photo_pickup_line")
        ):
            try:
                model_name = getattr(photo_generator, "model", "") or "paid model"
                if hasattr(photo_generator, "ensure_ready"):
                    photo_status = (
                        f"Profile {cycle}: generating a line from the first photo locally..."
                    )
                else:
                    photo_status = (
                        f"Profile {cycle}: generating a line from the first photo "
                        f"with {model_name}..."
                    )
                self.set_status(photo_status)
                photo_png = first_profile_photo_png(capture, heart_point)
                pickup_line = photo_generator.generate_photo_pickup_line(photo_png, tone)
                self.set_status(
                    f"Profile {cycle}: photo-grounded fallback ready: {pickup_line!r}"
                )
            except Exception as error:
                pickup_line = random_pickup_line()
                self.set_status(
                    f"Profile {cycle}: photo fallback unavailable ({error}); "
                    "using a safe built-in line."
                )
        elif pickup_line is None:
            pickup_line = random_pickup_line()

        # Re-sync after generation; paid API calls can take long enough for the
        # viewport to drift and hearts to leave the safe click band.
        current_capture, heart_match = self._capture_fallback_heart(scanner)
        if heart_match is None:
            return False
        heart_point, _ = heart_match
        for click_index in range(3):
            self.click_once(heart_point)
            if click_index < 2:
                self.interruptible_wait(0.12)
        self.interruptible_wait(0.85)

        comment_point = self.wait_for_comment_field(attempts=6)
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
            if send_point is None or send_score < SEND_LIKE_MIN_SCORE:
                return False
            self.set_status(
                f"Profile {cycle}: sending pickup-line {detected_text} "
                f"({send_attempt}/3)..."
            )
            self.click_once(send_point)
            _before_lines = dialog_lines
            _before_frame = getattr(dialog_capture, "frame", None)

            def classify_fallback(capture, lines):
                after_send, _, _ = find_send_priority_like(capture, lines)
                if after_send is None:
                    return "succeeded"
                if not reply_is_visible_near(
                    lines,
                    pickup_line,
                    comment_point,
                    dialog_capture.window.height,
                ):
                    return "succeeded"
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
        engine=PAID_ENGINE,
        paid_model=PAY_MODEL,
        api_key="",
        save_transcripts=False,
    ):
        started_at = time.monotonic()
        completed = 0
        fallback_likes = 0
        failures = []
        attempted = 0
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._save_transcripts = bool(save_transcripts)
        self._transcript_log_path = None
        try:
            if engine == "Local — Free":
                generator = OllamaReplyGenerator()
            else:
                choice = paid_model_from_selection(paid_model)
                if choice is None or choice.provider == "openai":
                    generator = ReplyGenerator(
                        model=(choice.model_id if choice else paid_model or PAY_MODEL),
                        api_key=api_key,
                    )
                else:
                    generator = CloudReplyGenerator(choice, api_key=api_key)
            model_name = getattr(generator, "model", "") or ""
            if self._save_transcripts:
                self._transcript_log_path = prompt_transcript_path(model=model_name)
                try:
                    header_path = Path(self._transcript_log_path)
                    header_path.parent.mkdir(parents=True, exist_ok=True)
                    with header_path.open("w", encoding="utf-8") as log_file:
                        log_file.write(
                            format_prompt_transcript_header(
                                batch_id=batch_id,
                                tone=tone,
                                engine=engine,
                                model=model_name,
                            )
                        )
                except OSError:
                    pass
            if hasattr(generator, "ensure_ready"):
                self.set_status("Warming up the local reply model...")
                generator.ensure_ready()
            scanner = self.make_profile_scanner(generator)
            rewind_next = False
            for cycle in range(1, self.total_rotations + 1):
                if not self.is_running:
                    break
                attempted += 1
                self._prompt_transcript = {}
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
                    if self.is_running and self._prompt_failure_can_skip:
                        if engine == "Local — Free":
                            should_fallback = True
                        elif is_paid_engine(engine):
                            # A paid photo fallback is safe only when scanning
                            # found no written prompt. Once a prompt was found,
                            # rewinding and choosing the topmost heart would
                            # redirect the reply to the first photo.
                            should_fallback = paid_photo_fallback_enabled(
                                engine,
                                self._prompt_stage,
                                error,
                            )
                        else:
                            should_fallback = False
                        if should_fallback:
                            with native_autorelease_pool():
                                fallback_sent = self.fallback_regular_like(
                                    scanner,
                                    cycle,
                                    photo_generator=generator,
                                    tone=tone,
                                    use_photo_generator=True,
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
                        self._prompt_stage,
                        error,
                        skipped,
                        recovery,
                    )
                    self.log_prompt_transcript(
                        "failed",
                        batch_id,
                        cycle,
                        tone,
                        engine,
                        model_name,
                        error=error,
                        stage=record["stage"],
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
                self.log_prompt_transcript(
                    "sent",
                    batch_id,
                    cycle,
                    tone,
                    engine,
                    model_name,
                )
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
                "transcript_log_path": (
                    str(self._transcript_log_path)
                    if self._transcript_log_path is not None
                    else None
                ),
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
        choice = self._selected_paid_model()
        paid_model = choice.label if choice else ""
        api_key = self.api_key_var.get().strip()
        if workflow == "Prompt Reply" and platform != "Hinge":
            self.status_label.config(text="Prompt Reply mode currently supports Hinge only.")
            return
        if workflow == "Prompt Reply" and is_paid_engine(engine) and choice is None:
            self.status_label.config(text="Select an available paid model.")
            return
        if workflow == "Prompt Reply" and is_paid_engine(engine) and not api_key:
            self.status_label.config(
                text=f"Import or paste your {choice.provider_title} API key."
            )
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
            engine_label = (
                f"{choice.provider_title} ({choice.label})"
                if is_paid_engine(engine) and choice is not None
                else engine
            )
            self.render_status(
                f"Starting Hinge Prompt Reply with {tone} tone using {engine_label}..."
            )
            target = self.run_prompt_reply
            arguments = (
                tone,
                engine,
                paid_model,
                api_key,
                bool(self.save_transcript_var.get()),
            )
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
