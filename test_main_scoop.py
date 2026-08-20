import unittest
from contextlib import contextmanager
import os
import struct
import tempfile
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from pynput.keyboard import Key
from types import SimpleNamespace
from unittest.mock import patch

from main_scoop import (
    HeartIconDetector,
    HINGE_HEART_BUTTON_TEMPLATE,
    MirroringWindow,
    PAID_MODEL_PROMPT,
    PixelFrame,
    SCREEN_RECORDING_SETTINGS_URL,
    ScoopUpApp,
    WindowCapture,
    _window_score,
    find_send_priority_like,
    format_elapsed_time,
    is_safe_iphone_action_point,
    first_profile_photo_png,
    parse_openai_api_key,
    prompt_transcript_path,
)
from profile_reply import CapturedPrompt, ProfileScan, ProfileScanError, ScreenText, normalize_text
from reply_generation import GeneratedReply, PAID_ENGINE, PAY_MODELS


def make_frame(width, height, colored_points, color):
    pixels = bytearray((245, 245, 245, 255) * (width * height))
    for x, y in colored_points:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 4
            pixels[offset:offset + 3] = bytes(color)
    return PixelFrame(width, height, width * 4, bytes(pixels))


def make_capture(frame, retina_scale=1):
    window = MirroringWindow(
        window_id=7,
        owner_pid=1,
        owner_name="iPhone Mirroring",
        window_name="iPhone",
        left=100,
        top=50,
        width=frame.width / retina_scale,
        height=frame.height / retina_scale,
    )
    return WindowCapture(window, None, frame)


class AutomaticDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = HeartIconDetector()

    def test_filled_and_outline_hearts_score_high(self):
        filled = list(self.detector.expected_fill)
        outline = list(self.detector.expected_boundary)
        self.assertGreater(self.detector.shape_score(filled, (0, 0, 24, 24)), 0.9)
        self.assertGreater(self.detector.shape_score(outline, (0, 0, 24, 24)), 0.9)

    def test_rectangle_is_rejected_as_heart(self):
        rectangle = [(x, y) for y in range(16) for x in range(24)]
        self.assertLess(self.detector.shape_score(rectangle, (0, 0, 24, 16)), 0.74)

    def test_tinder_heart_is_found_without_fixed_coordinates(self):
        offset_x, offset_y = 80, 190
        points = {
            (offset_x + x, offset_y + y)
            for x, y in self.detector.expected_fill
        }
        capture = make_capture(make_frame(220, 320, points, (25, 205, 35)))
        location, score = self.detector.find(capture, "Tinder")
        self.assertLess(abs(location[0] - (100 + offset_x + 12)), 2)
        self.assertLess(abs(location[1] - (50 + offset_y + 12)), 2)
        self.assertGreater(score, 0.66)

    def test_hinge_black_circle_white_heart_is_found_at_variable_position(self):
        offset_x, offset_y = 165, 120
        icon_size = 72
        points = set()
        for y in range(icon_size):
            for x in range(icon_size):
                row = min(23, int(y / icon_size * 24))
                column = min(23, int(x / icon_size * 24))
                if HINGE_HEART_BUTTON_TEMPLATE[row][column] == "#":
                    points.add((offset_x + x, offset_y + y))
        capture = make_capture(make_frame(240, 360, points, (25, 25, 25)))
        location, score = self.detector.find(capture, "Hinge")
        self.assertLessEqual(abs(location[0] - (100 + offset_x + icon_size / 2)), 2)
        self.assertLessEqual(abs(location[1] - (50 + offset_y + icon_size / 2)), 2)
        self.assertGreater(score, 0.84)

    def test_hinge_heart_is_found_against_bottom_right_edge(self):
        frame_width, frame_height = 240, 360
        icon_size = 72
        offset_x = frame_width - icon_size
        offset_y = frame_height - icon_size
        points = set()
        for y in range(icon_size):
            for x in range(icon_size):
                row = min(23, int(y / icon_size * 24))
                column = min(23, int(x / icon_size * 24))
                if HINGE_HEART_BUTTON_TEMPLATE[row][column] == "#":
                    points.add((offset_x + x, offset_y + y))

        capture = make_capture(make_frame(frame_width, frame_height, points, (25, 25, 25)))
        location, score = self.detector.find(capture, "Hinge")

        self.assertIsNotNone(location)
        self.assertGreater(score, 0.84)

    def test_hinge_heart_is_found_in_retina_capture(self):
        frame_width, frame_height = 480, 720
        icon_size = 72
        offset_x, offset_y = 360, 560
        points = set()
        for y in range(icon_size):
            for x in range(icon_size):
                row = min(23, int(y / icon_size * 24))
                column = min(23, int(x / icon_size * 24))
                if HINGE_HEART_BUTTON_TEMPLATE[row][column] == "#":
                    points.add((offset_x + x, offset_y + y))

        capture = make_capture(
            make_frame(frame_width, frame_height, points, (25, 25, 25)),
            retina_scale=2,
        )
        location, score = self.detector.find(capture, "Hinge")

        self.assertAlmostEqual(location[0], 100 + (offset_x + icon_size / 2) / 2, delta=1)
        self.assertAlmostEqual(location[1], 50 + (offset_y + icon_size / 2) / 2, delta=1)
        self.assertGreater(score, 0.84)

    def test_hinge_template_fallback_works_when_button_merges_with_dark_photo(self):
        frame_width, frame_height = 480, 720
        icon_size = 72
        offset_x, offset_y = 360, 560
        # A large dark photo region touches the button's left edge, making one
        # oversized connected component that the shape pass must reject.
        points = {
            (x, y)
            for y in range(500, 650)
            for x in range(250, offset_x + 1)
        }
        for y in range(icon_size):
            for x in range(icon_size):
                row = min(23, int(y / icon_size * 24))
                column = min(23, int(x / icon_size * 24))
                if HINGE_HEART_BUTTON_TEMPLATE[row][column] == "#":
                    points.add((offset_x + x, offset_y + y))

        capture = make_capture(
            make_frame(frame_width, frame_height, points, (25, 25, 25)),
            retina_scale=2,
        )
        location, score = self.detector.find(capture, "Hinge")

        self.assertIsNotNone(location)
        self.assertGreater(score, 0.84)

    def test_first_profile_photo_is_cropped_to_bounded_png(self):
        capture = make_capture(make_frame(400, 800, [], (30, 80, 160)))
        image = first_profile_photo_png(capture, (300, 650), max_edge=256)

        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", image[16:24])
        self.assertLessEqual(max(width, height), 256)
        self.assertGreater(width, 80)
        self.assertGreater(height, 80)

    def test_plain_black_circle_is_not_the_hinge_heart_button(self):
        size = 72
        circle = [
            (x, y)
            for y in range(size)
            for x in range(size)
            if (x - 35.5) ** 2 + (y - 35.5) ** 2 <= 35 ** 2
        ]
        frame = make_frame(size, size, circle, (25, 25, 25))
        self.assertLess(
            self.detector.hinge_patch_score(frame, 0, 0, size),
            0.84,
        )

    def test_iphone_mirroring_bundle_is_preferred(self):
        score = _window_score(
            "Localized App Name",
            "Phone",
            "com.apple.ScreenContinuity",
            430,
            850,
        )
        self.assertGreaterEqual(score, 200)
        self.assertLess(_window_score("iPhone Simulator", "", "", 430, 850), 0)

    def test_send_like_text_normalization(self):
        self.assertEqual(normalize_text("Send Priority Like!"), "send priority like")

    def test_elapsed_time_is_compact_and_readable(self):
        self.assertEqual(format_elapsed_time(8.4), "8s")
        self.assertEqual(format_elapsed_time(128), "2m 8s")
        self.assertEqual(format_elapsed_time(3723), "1h 2m 3s")

    def test_action_guard_rejects_home_area(self):
        capture = make_capture(make_frame(400, 800, [], (0, 0, 0)))
        self.assertTrue(is_safe_iphone_action_point(capture, (300, 650)))
        self.assertFalse(is_safe_iphone_action_point(capture, (300, 770)))

    def test_dialog_send_fallback_rejects_home_area(self):
        app = object.__new__(ScoopUpApp)
        capture = make_capture(make_frame(400, 800, [], (0, 0, 0)))
        comment_point = (300, 735)
        with patch("main_scoop.find_send_priority_like", return_value=(None, 0.0, "")):
            point, score, _ = app.find_send_for_dialog(capture, [], comment_point)
        self.assertIsNone(point)
        self.assertEqual(score, 0.0)

    def test_screen_recording_settings_uses_privacy_deep_link(self):
        self.assertTrue(SCREEN_RECORDING_SETTINGS_URL.startswith("x-apple.systempreferences:"))
        self.assertIn("Privacy_ScreenCapture", SCREEN_RECORDING_SETTINGS_URL)

    def test_confirmed_target_is_clicked_three_times(self):
        class FakeMouse:
            def __init__(self):
                self.position = None
                self.clicks = 0

            def click(self, _button, count):
                self.clicks += count

        app = object.__new__(ScoopUpApp)
        app.mouse = FakeMouse()
        app.is_running = True
        app.click_repeatedly((123, 456), interval=0)

        self.assertEqual(app.mouse.position, (123, 456))
        self.assertEqual(app.mouse.clicks, 3)

    def test_heart_scan_is_single_flight_and_drains_native_pool(self):
        app = object.__new__(ScoopUpApp)
        app.fresh_capture = lambda: object()
        app.heart_detector = SimpleNamespace(
            find=lambda _capture, _platform: ((123, 456), 0.91)
        )
        pool_events = []

        @contextmanager
        def tracked_pool():
            pool_events.append("enter")
            try:
                yield
            finally:
                pool_events.append("exit")

        with (
            patch("main_scoop.native_autorelease_pool", tracked_pool),
            patch("main_scoop.threading.Thread") as thread_class,
        ):
            result = app.find_heart_before("Hinge", time.monotonic() + 1)

        self.assertEqual(result, ((123, 456), 0.91, False))
        self.assertEqual(pool_events, ["enter", "exit"])
        thread_class.assert_not_called()

    def test_profile_scroll_uses_negative_pixel_events_for_next_viewport(self):
        class FakeMouse:
            def __init__(self):
                self.positions = []

            @property
            def position(self):
                return self.positions[-1] if self.positions else None

            @position.setter
            def position(self, value):
                self.positions.append(value)

        class FakeRunningApplication:
            def __init__(self):
                self.activations = []

            def activateWithOptions_(self, options):
                self.activations.append(options)

        running_app = FakeRunningApplication()
        fake_appkit = SimpleNamespace(
            NSApplicationActivateIgnoringOtherApps=99,
            NSRunningApplication=SimpleNamespace(
                runningApplicationWithProcessIdentifier_=lambda _pid: running_app
            ),
        )
        posted = []
        fake_quartz = SimpleNamespace(
            kCGScrollEventUnitPixel=1,
            kCGHIDEventTap=2,
            CGEventCreateScrollWheelEvent=lambda _source, _unit, _count, delta: delta,
            CGEventPost=lambda _tap, event: posted.append(event),
        )

        app = object.__new__(ScoopUpApp)
        app.mouse = FakeMouse()
        app.is_running = True
        window = MirroringWindow(7, 1, "iPhone Mirroring", "iPhone", 100, 50, 400, 800)

        with (
            patch("main_scoop.find_iphone_mirroring_window", return_value=window),
            patch("main_scoop.AppKit", fake_appkit),
            patch("main_scoop.Quartz", fake_quartz),
            patch("main_scoop.time.sleep"),
        ):
            app.scroll_profile("down")

        self.assertEqual(posted, [-140, -140])
        self.assertEqual(running_app.activations, [99])


class PromptReplyOrchestrationTests(unittest.TestCase):
    def make_app(self):
        app = object.__new__(ScoopUpApp)
        app.is_running = True
        app.total_rotations = 1
        app._prompt_stage = "unknown"
        app._prompt_failure_can_skip = True
        app.last_prompt_batch = None
        app._save_transcripts = False
        app._prompt_transcript = {}
        app._transcript_log_path = Path("/tmp/scoop-unused-transcripts.txt")
        app.root = SimpleNamespace(after=lambda _delay, callback: callback())
        app.start_button = SimpleNamespace(config=lambda **_kwargs: None)
        app.statuses = []
        app.set_status = lambda message, **_kwargs: app.statuses.append(message)
        app.interruptible_wait = lambda _seconds: None
        app.fresh_capture = lambda: SimpleNamespace(
            window=SimpleNamespace(left=0, top=0, width=400, height=800)
        )
        app.paste_reply = lambda _reply, defer_restore=False: None
        app.type_reply = lambda _reply: None
        app.clicks = []
        app.click_once = lambda point: app.clicks.append(point)
        app.fallback_regular_like = (
            lambda _scanner, _cycle, _line="", **_kwargs: False
        )
        app.skip_current_profile = lambda: True
        app.log_prompt_failure = lambda batch, cycle, stage, error, skipped, recovery=None: {
            "batch_id": batch,
            "rotation": cycle,
            "stage": stage,
            "message": str(error),
            "skipped": skipped,
            "recovery": recovery,
        }
        return app

    def make_scan(self):
        prompt = CapturedPrompt(
            "prompt-1",
            "Together, we could",
            "Win trivia night",
            0,
            0,
            (100, 200),
            0.95,
        )
        scan = ProfileScan((prompt,), 1, "fingerprint")
        scanner = SimpleNamespace(
            scan=lambda ensure_top=True: scan,
            relocate=lambda _target: prompt,
            reconfirm_visible=lambda _target: prompt,
            center_target=lambda _target, relocated: relocated,
        )
        generator = SimpleNamespace(
            generate=lambda _prompts, _tone: GeneratedReply(
                "prompt-1",
                "I bring facts and confidently wrong bonus-round guesses.",
            )
        )
        return scan, scanner, generator

    def test_closed_send_dialog_counts_as_success_when_prompt_card_remains_visible(self):
        app = self.make_app()
        scan, _scanner, _generator = self.make_scan()

        with (
            patch("main_scoop.find_send_priority_like", return_value=(None, 0.0, "")),
            patch("main_scoop.reply_is_visible_near", return_value=False),
            patch("main_scoop.prompt_is_visible", return_value=True),
        ):
            outcome = app._classify_prompt_send_outcome(
                object(),
                [],
                scan.prompts[0],
                "A grounded reply",
                (200, 500),
                800,
            )

        self.assertEqual(outcome, "succeeded")

    def test_closed_send_dialog_counts_as_success_when_reply_ocr_lingers(self):
        app = self.make_app()
        scan, _scanner, _generator = self.make_scan()

        with (
            patch("main_scoop.find_send_priority_like", return_value=(None, 0.0, "")),
            patch("main_scoop.reply_is_visible_near", return_value=True),
            patch("main_scoop.prompt_is_visible", return_value=True),
        ):
            outcome = app._classify_prompt_send_outcome(
                object(),
                [],
                scan.prompts[0],
                "A grounded reply",
                (200, 500),
                800,
            )

        self.assertEqual(outcome, "succeeded")

    def test_comment_field_uses_send_button_geometry_when_placeholder_is_missed(self):
        app = self.make_app()
        capture = SimpleNamespace(
            window=SimpleNamespace(left=100, top=50, width=400, height=700)
        )
        app.fresh_capture = lambda: capture
        send_line = ScreenText("Send Priority Like", 0.96, 250, 620, 180, 30)

        with patch("main_scoop.recognize_text", return_value=[send_line]):
            point = app.wait_for_comment_field()

        self.assertEqual(point, (300, send_line.center[1] - 59.5))

    def test_five_profile_batch_drains_native_pool_after_every_profile(self):
        app = self.make_app()
        app.total_rotations = 5
        app.make_profile_scanner = lambda _generator=None: object()
        processed = []
        pool_events = []

        def process(_scanner, _generator, _tone, cycle, ensure_top):
            processed.append((cycle, ensure_top))
            # Stand in for temporary screenshot and OCR working buffers.
            temporary_pixels = bytearray(8 * 1024 * 1024)
            return f"reply-{cycle}-{len(temporary_pixels)}"

        app._process_prompt_reply_profile = process

        @contextmanager
        def tracked_pool():
            pool_events.append("enter")
            try:
                yield
            finally:
                pool_events.append("exit")

        with (
            patch("main_scoop.ReplyGenerator", return_value=object()),
            patch("main_scoop.native_autorelease_pool", tracked_pool),
        ):
            app.run_prompt_reply("Playful & clean")

        self.assertEqual(
            processed,
            [(1, True), (2, False), (3, False), (4, False), (5, False)],
        )
        self.assertEqual(pool_events, ["enter", "exit"] * 5)
        self.assertEqual(app.last_prompt_batch["completed"], 5)

    def test_failed_prompt_uses_qwen_first_photo_before_clicking_fallback(self):
        app = self.make_app()
        capture = make_capture(make_frame(400, 800, [], (25, 100, 180)))
        heart_point = (300, 600)
        app.fresh_capture = lambda: capture
        app.heart_detector = SimpleNamespace(
            find_all=lambda _capture, _platform: [(heart_point, 0.95)]
        )
        app.wait_for_comment_field = lambda attempts=3: None
        scanner = SimpleNamespace(scroll_to_top=lambda: None)
        photo_calls = []
        generator = SimpleNamespace(
            generate_photo_pickup_line=lambda image, tone: (
                photo_calls.append((image, tone))
                or "Does that surfboard come with lessons, or just confidence?"
            )
        )

        with (
            patch("main_scoop.find_send_priority_like", return_value=(None, 0.0, "")),
            patch("main_scoop.viewport_similarity", return_value=1.0),
        ):
            sent = ScoopUpApp.fallback_regular_like(
                app,
                scanner,
                1,
                photo_generator=generator,
                tone="Playful & clean",
            )

        self.assertFalse(sent)
        self.assertEqual(len(photo_calls), 1)
        self.assertTrue(photo_calls[0][0].startswith(b"\x89PNG"))
        self.assertEqual(photo_calls[0][1], "Playful & clean")
        self.assertEqual(app.clicks.count(heart_point), 3)

    def test_fallback_skips_photo_when_use_photo_generator_false(self):
        app = self.make_app()
        capture = make_capture(make_frame(400, 800, [], (25, 100, 180)))
        heart_point = (300, 600)
        app.fresh_capture = lambda: capture
        app.heart_detector = SimpleNamespace(
            find_all=lambda _capture, _platform: [(heart_point, 0.95)]
        )
        app.wait_for_comment_field = lambda attempts=3: None
        scanner = SimpleNamespace(scroll_to_top=lambda: None)
        photo_calls = []
        generator = SimpleNamespace(
            generate_photo_pickup_line=lambda image, tone: photo_calls.append((image, tone)),
        )

        with (
            patch("main_scoop.find_send_priority_like", return_value=(None, 0.0, "")),
            patch("main_scoop.viewport_similarity", return_value=1.0),
        ):
            ScoopUpApp.fallback_regular_like(
                app,
                scanner,
                1,
                photo_generator=generator,
                tone="Playful & clean",
                use_photo_generator=False,
            )

        self.assertEqual(photo_calls, [])

    def test_fallback_hearts_include_low_photo_hearts(self):
        app = self.make_app()
        capture = make_capture(make_frame(400, 800, [], (25, 100, 180)))
        low_photo_heart = (300, 744)
        app.heart_detector = SimpleNamespace(
            find_all=lambda _capture, _platform: [(low_photo_heart, 0.88)]
        )
        self.assertEqual(app._hinge_hearts_in_band(capture), [])
        self.assertEqual(len(app._fallback_hearts_in_band(capture)), 1)

    def test_paid_scan_failure_enables_photo_fallback(self):
        app = self.make_app()
        app.make_profile_scanner = lambda _generator=None: object()
        captured = {}

        def fallback(*_args, **kwargs):
            captured.update(kwargs)
            return False

        def fail_scan(*_args, **_kwargs):
            app._prompt_stage = "scan"
            raise ProfileScanError(
                "No readable Hinge prompt and answer pairs were found."
            )

        app._process_prompt_reply_profile = fail_scan
        app.fallback_regular_like = fallback
        with patch(
            "main_scoop.ReplyGenerator",
            return_value=SimpleNamespace(model="gpt-5-mini"),
        ):
            app.run_prompt_reply("Playful & clean", engine="Paid API")

        self.assertTrue(captured.get("use_photo_generator"))

    def test_paid_send_failure_skips_photo_fallback(self):
        app = self.make_app()
        app.make_profile_scanner = lambda _generator=None: object()
        captured = {}

        def fallback(*_args, **kwargs):
            captured.update(kwargs)
            return False

        def fail_send(*_args, **_kwargs):
            app._prompt_stage = "enter_reply"
            raise ProfileScanError(
                "The entered reply could not be verified; nothing was sent."
            )

        app._process_prompt_reply_profile = fail_send
        app.fallback_regular_like = fallback
        with patch(
            "main_scoop.ReplyGenerator",
            return_value=SimpleNamespace(model="gpt-5-mini"),
        ):
            app.run_prompt_reply("Playful & clean", engine="Paid API")

        self.assertFalse(captured.get("use_photo_generator"))

    def test_prompt_heart_hands_off_when_composer_starts_offscreen(self):
        app = self.make_app()
        _scan, scanner, _generator = self.make_scan()
        selected = scanner.scan().prompts[0]
        app.wait_for_comment_field = lambda attempts=2: None
        scanner.reconfirm_visible = lambda _prompt: self.fail(
            "prompt must not be re-detected after the heart click"
        )

        point = app.open_prompt_dialog(scanner, selected, selected)

        self.assertIsNone(point)
        self.assertEqual(app.clicks.count(selected.heart_point), 3)

    def test_ascii_local_reply_is_direct_typed_until_composer_verifies(self):
        app = self.make_app()
        attempts = []
        app.paste_reply = attempts.append
        typed = []
        app.type_reply = typed.append
        capture = SimpleNamespace(window=SimpleNamespace(height=800))
        app.verify_reply_entry = lambda *_args, **_kwargs: (capture, [])

        with patch(
            "main_scoop.reply_is_visible_near",
            side_effect=[False, False, True],
        ):
            result_capture, _ = app.enter_reply(
                (200, 500),
                "A grounded reply",
                800,
            )

        self.assertIs(result_capture, capture)
        self.assertEqual(attempts, [])
        self.assertEqual(typed, ["A grounded reply"] * 3)

    def test_paid_prepared_reply_is_direct_typed_until_composer_verifies(self):
        app = self.make_app()
        pasted = []
        app.paste_reply = lambda reply, defer_restore=False: pasted.append(reply)
        typed = []
        app.type_reply = typed.append
        capture = SimpleNamespace(window=SimpleNamespace(height=800))
        app.verify_reply_entry = lambda *_args, **_kwargs: (capture, [])

        with patch(
            "main_scoop.reply_is_visible_near",
            side_effect=[False, True],
        ):
            result_capture, _ = app.enter_reply(
                (200, 500),
                "Trivia rivals first",
                800,
            )

        self.assertIs(result_capture, capture)
        self.assertEqual(pasted, [])
        self.assertEqual(typed, ["Trivia rivals first"] * 2)

    def test_reply_verification_uses_short_waits_only_between_attempts(self):
        app = self.make_app()
        waits = []
        app.interruptible_wait = waits.append
        capture = SimpleNamespace(window=SimpleNamespace(height=800))
        app.fresh_capture = lambda: capture

        with (
            patch("main_scoop.recognize_text", return_value=[]),
            patch("main_scoop.reply_is_visible_near", return_value=False),
        ):
            app.verify_reply_entry(
                "A grounded reply",
                (200, 500),
                800,
                attempts=3,
            )

        self.assertEqual(waits, [0.15, 0.15])

    def test_failed_text_verification_never_clicks_send(self):
        app = self.make_app()
        _scan, scanner, generator = self.make_scan()
        app.make_profile_scanner = lambda _generator=None: scanner
        comment_line = ScreenText("Add a comment", 0.95, 100, 430, 200, 40)
        send_point = (250, 520)

        with (
            patch("main_scoop.ReplyGenerator", return_value=generator),
            patch("main_scoop.recognize_text", return_value=[comment_line]),
            patch("main_scoop.prompt_is_visible", return_value=True),
            patch("main_scoop.reply_is_visible_near", return_value=False),
            patch(
                "main_scoop.find_send_priority_like",
                return_value=(send_point, 0.95, "Send Like"),
            ),
        ):
            app.run_prompt_reply("Playful & clean")

        # Send detection is allowed for safe layout positioning, but no Send
        # coordinate may be clicked when reply verification fails.
        self.assertNotIn(send_point, app.clicks)
        self.assertEqual(
            app.clicks,
            [
                (100, 200),
                (100, 200),
                (100, 200),
                comment_line.center,
                comment_line.center,
                comment_line.center,
                comment_line.center,
                comment_line.center,
                comment_line.center,
            ],
        )
        self.assertTrue(any("could not be verified" in status for status in app.statuses))

    def test_unresponsive_send_is_retried_three_times_only_while_state_matches(self):
        app = self.make_app()
        _scan, scanner, generator = self.make_scan()
        app.make_profile_scanner = lambda _generator=None: scanner
        comment_line = ScreenText("Add a comment", 0.95, 10, 10, 100, 20)
        entered_line = ScreenText("Verified funny reply", 0.95, 10, 50, 180, 20)
        send_point = (250, 520)

        with (
            patch("main_scoop.ReplyGenerator", return_value=generator),
            patch(
                "main_scoop.recognize_text",
                return_value=[comment_line, entered_line],
            ),
            patch("main_scoop.prompt_is_visible", return_value=True),
            patch("main_scoop.reply_is_visible_near", return_value=True),
            patch(
                "main_scoop.find_send_priority_like",
                return_value=(send_point, 0.95, "Send Like"),
            ),
        ):
            app.run_prompt_reply("Playful & clean")

        self.assertEqual(app.clicks.count(send_point), 3)
        self.assertTrue(any("after 3 verified" in status for status in app.statuses))

    def test_mismatched_open_dialog_never_pastes_or_sends(self):
        app = self.make_app()
        _scan, scanner, generator = self.make_scan()
        app.make_profile_scanner = lambda _generator=None: scanner
        pasted = []
        app.paste_reply = pasted.append
        comment_line = ScreenText("Add a comment", 0.95, 100, 430, 200, 40)
        send_line = ScreenText("Send Like", 0.95, 220, 500, 160, 28)
        send_point = send_line.center

        with (
            patch("main_scoop.ReplyGenerator", return_value=generator),
            patch(
                "main_scoop.recognize_text",
                return_value=[comment_line, send_line],
            ),
            patch("main_scoop.prompt_is_visible", return_value=False),
        ):
            app.run_prompt_reply("Playful & clean")

        self.assertEqual(pasted, [])
        self.assertNotIn(send_point, app.clicks)
        self.assertTrue(any("did not match" in status for status in app.statuses))

    def test_find_send_priority_like_reuses_provided_lines(self):
        send_line = ScreenText("Send Like", 0.95, 220, 500, 160, 28)
        capture = SimpleNamespace()
        with patch("main_scoop.recognize_text") as ocr:
            point, score, text = find_send_priority_like(capture, [send_line])

        ocr.assert_not_called()
        self.assertEqual(point, send_line.center)
        self.assertEqual(text, "Send Like")
        self.assertGreaterEqual(score, 0.95)

    def test_open_dialog_scrolls_until_send_like_reaches_top_eighty_five_percent(self):
        app = self.make_app()
        scrolls = []
        waits = []
        app.interruptible_wait = waits.append
        app.scroll_profile = lambda direction: scrolls.append(direction)
        comment_line = ScreenText("Add a comment", 0.95, 100, 430, 200, 40)

        def recognize(_capture, _vision, accurate=True):
            send_top = 736 - len(scrolls) * 100
            return [
                comment_line,
                ScreenText("Send Like", 0.95, 220, send_top, 160, 28),
            ]

        with patch("main_scoop.recognize_text", side_effect=recognize):
            _capture, _lines, comment_point = app.position_open_dialog_safely()

        self.assertEqual(comment_point, comment_line.center)
        self.assertEqual(scrolls, ["down_small"])
        self.assertEqual(waits, [0.18])

    def test_prompt_heart_burst_captures_once_before_clicks(self):
        app = self.make_app()
        _scan, scanner, _generator = self.make_scan()
        selected = scanner.scan().prompts[0]
        captures = []
        capture = SimpleNamespace(
            window=SimpleNamespace(left=0, top=0, width=400, height=800)
        )
        app.fresh_capture = lambda: captures.append(capture) or capture
        app.wait_for_comment_field = lambda attempts=2: (300, 400)

        point = app.open_prompt_dialog(scanner, selected, selected)

        self.assertEqual(point, (300, 400))
        self.assertEqual(len(captures), 1)
        self.assertEqual(app.clicks.count(selected.heart_point), 3)

    def test_prompt_heart_is_moved_only_when_below_safe_click_area(self):
        app = self.make_app()
        _scan, scanner, _generator = self.make_scan()
        selected = scanner.scan().prompts[0]
        unsafe = CapturedPrompt(
            selected.prompt_id,
            selected.prompt,
            selected.answer,
            selected.viewport_index,
            selected.scroll_steps,
            (selected.heart_point[0], 700),
            selected.confidence,
        )
        moved = CapturedPrompt(
            selected.prompt_id,
            selected.prompt,
            selected.answer,
            selected.viewport_index,
            selected.scroll_steps,
            (selected.heart_point[0], 500),
            selected.confidence,
        )
        adjustments = []
        scanner.center_target = (
            lambda target, current: adjustments.append((target, current)) or moved
        )
        app.wait_for_comment_field = lambda attempts=2: (300, 400)

        point = app.open_prompt_dialog(scanner, selected, unsafe)

        self.assertEqual(point, (300, 400))
        self.assertEqual(adjustments, [(selected, unsafe)])
        self.assertEqual(app.clicks.count(unsafe.heart_point), 0)
        self.assertEqual(app.clicks.count(moved.heart_point), 3)

    def test_send_succeeds_as_soon_as_dialog_disappears(self):
        app = self.make_app()
        _scan, scanner, generator = self.make_scan()
        app.make_profile_scanner = lambda _generator=None: scanner
        waits = []
        app.interruptible_wait = lambda seconds: waits.append(seconds)
        comment_line = ScreenText("Add a comment", 0.95, 100, 430, 200, 40)
        entered_line = ScreenText(
            "I bring facts and confidently wrong bonus-round guesses.",
            0.95,
            10,
            50,
            180,
            20,
        )
        send_line = ScreenText("Send Like", 0.95, 220, 500, 160, 28)
        send_point = send_line.center

        def recognize(_capture, _vision, accurate=True):
            if send_point in app.clicks:
                return []
            return [comment_line, entered_line, send_line]

        with (
            patch("main_scoop.ReplyGenerator", return_value=generator),
            patch("main_scoop.recognize_text", side_effect=recognize),
            patch(
                "main_scoop.prompt_is_visible",
                side_effect=lambda _lines, _selected: send_point not in app.clicks,
            ),
            patch(
                "main_scoop.reply_is_visible_near",
                side_effect=lambda *_args, **_kwargs: send_point not in app.clicks,
            ),
        ):
            app.run_prompt_reply("Playful & clean")

        self.assertEqual(app.clicks.count(send_point), 1)
        self.assertNotIn(1.1, waits)
        self.assertTrue(any("sent" in status.lower() for status in app.statuses))

    def test_prompt_heart_is_clicked_before_generation_finishes(self):
        app = self.make_app()
        _scan, scanner, _generator = self.make_scan()
        selected = scanner.scan().prompts[0]
        app.make_profile_scanner = lambda _generator=None: scanner
        order = []
        started = threading.Event()
        release = threading.Event()

        def generate(_prompts, _tone):
            order.append("generate_start")
            started.set()
            self.assertTrue(release.wait(1))
            order.append("generate_end")
            return GeneratedReply(
                "prompt-1",
                "I bring facts and confidently wrong bonus-round guesses.",
            )

        generator = SimpleNamespace(generate=generate)
        comment_line = ScreenText("Add a comment", 0.95, 100, 430, 200, 40)
        entered_line = ScreenText(
            "I bring facts and confidently wrong bonus-round guesses.",
            0.95,
            10,
            50,
            180,
            20,
        )
        send_line = ScreenText("Send Like", 0.95, 220, 500, 160, 28)
        send_point = send_line.center

        def click(point):
            if point == selected.heart_point:
                order.append("heart")
                release.set()
            app.clicks.append(point)

        def recognize(_capture, _vision, accurate=True):
            if send_point in app.clicks:
                return []
            return [comment_line, entered_line, send_line]

        app.click_once = click
        with (
            patch("main_scoop.ReplyGenerator", return_value=generator),
            patch("main_scoop.recognize_text", side_effect=recognize),
            patch(
                "main_scoop.prompt_is_visible",
                side_effect=lambda _lines, _selected: send_point not in app.clicks,
            ),
            patch(
                "main_scoop.reply_is_visible_near",
                side_effect=lambda *_args, **_kwargs: send_point not in app.clicks,
            ),
        ):
            app.run_prompt_reply("Playful & clean")

        self.assertEqual(order[0], "generate_start")
        self.assertLess(order.index("heart"), order.index("generate_end"))
        self.assertEqual(app.clicks.count(send_point), 1)

    def test_offscreen_composer_still_sends_after_heart_click(self):
        app = self.make_app()
        _scan, scanner, generator = self.make_scan()
        app.make_profile_scanner = lambda _generator=None: scanner
        app.open_prompt_dialog = lambda *_args, **_kwargs: None
        comment_line = ScreenText("Add a comment", 0.95, 100, 430, 200, 40)
        entered_line = ScreenText(
            "I bring facts and confidently wrong bonus-round guesses.",
            0.95,
            10,
            50,
            180,
            20,
        )
        send_line = ScreenText("Send Like", 0.95, 220, 500, 160, 28)
        send_point = send_line.center

        def recognize(_capture, _vision, accurate=True):
            if send_point in app.clicks:
                return []
            return [comment_line, entered_line, send_line]

        with (
            patch("main_scoop.ReplyGenerator", return_value=generator),
            patch("main_scoop.recognize_text", side_effect=recognize),
            patch("main_scoop.prompt_is_visible", return_value=True),
            patch(
                "main_scoop.reply_is_visible_near",
                side_effect=lambda *_args, **_kwargs: send_point not in app.clicks,
            ),
        ):
            app.run_prompt_reply("Playful & clean")

        self.assertEqual(app.clicks.count(send_point), 1)
        self.assertTrue(any("sent" in status.lower() for status in app.statuses))

    def test_batch_continues_all_rotations_when_skip_fails(self):
        app = self.make_app()
        app.total_rotations = 6
        processed = []
        app.make_profile_scanner = lambda _generator=None: object()
        app.skip_current_profile = lambda: False

        def process(_scanner, _generator, _tone, cycle, ensure_top):
            processed.append((cycle, ensure_top))
            raise ProfileScanError("prompt heart was covered by the like sheet")

        app._process_prompt_reply_profile = process

        with patch("main_scoop.ReplyGenerator", return_value=object()):
            app.run_prompt_reply("Playful & clean")

        self.assertEqual(
            processed,
            [(1, True), (2, True), (3, True), (4, True), (5, True), (6, True)],
        )
        self.assertEqual(app.last_prompt_batch["attempted"], 6)
        self.assertEqual(app.last_prompt_batch["completed"], 0)

    def test_transcript_log_stays_off_by_default(self):
        app = self.make_app()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Scoop unused.txt"
            app.make_profile_scanner = lambda _generator=None: object()
            app._process_prompt_reply_profile = lambda *_args, **_kwargs: "Trivia first?"
            with (
                patch("main_scoop.ReplyGenerator", return_value=SimpleNamespace(model="gpt-5-mini")),
                patch("main_scoop.prompt_transcript_path", return_value=path),
            ):
                app.run_prompt_reply("Playful & clean")
            self.assertIsNone(app._transcript_log_path)
            self.assertFalse(path.exists())

    def test_transcript_log_writes_sent_and_failed_profiles(self):
        app = self.make_app()
        app.total_rotations = 2
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Scoop gpt-5-mini 2026-08-19 18-45-30.txt"
            app.make_profile_scanner = lambda _generator=None: object()

            def process(_scanner, _generator, _tone, cycle, ensure_top):
                app._prompt_transcript = {
                    "profile_prompt": "Together, we could",
                    "profile_answer": "Win trivia night",
                    "model_input": (
                        "Generate a Playful & clean reply for this dating profile:\n"
                        "Together, we could\n"
                        "Win trivia night"
                    ),
                    "model_reply": "Trivia night sounds like a plan.",
                }
                if cycle == 2:
                    raise ProfileScanError("The Send Like control was not confidently detected.")
                return app._prompt_transcript["model_reply"]

            app._process_prompt_reply_profile = process
            with (
                patch("main_scoop.ReplyGenerator", return_value=SimpleNamespace(model="gpt-5-mini")),
                patch("main_scoop.prompt_transcript_path", return_value=path),
            ):
                app.run_prompt_reply("Playful & clean", save_transcripts=True)

            self.assertEqual(app._transcript_log_path, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("The Scoop UP Prompt Reply Log", text)
            self.assertIn("Engine: Paid API", text)
            self.assertIn("Model: gpt-5-mini", text)
            self.assertIn("Tone: Playful & clean", text)
            self.assertEqual(text.count("Engine:"), 1)
            self.assertEqual(text.count("Model: gpt-5-mini"), 1)
            self.assertEqual(text.count("Tone: Playful & clean"), 1)
            self.assertIn("Profile 1", text)
            self.assertIn("  sent  ", text)
            self.assertIn("Together, we could", text)
            self.assertIn("Win trivia night", text)
            self.assertIn(
                "Sent to model:\n"
                "Generate a Playful & clean reply for this dating profile:\n"
                "Together, we could\n"
                "Win trivia night",
                text,
            )
            self.assertIn("Reply sent to profile:\nTrivia night sounds like a plan.", text)
            self.assertIn("Profile 2", text)
            self.assertIn("  failed  ", text)
            self.assertIn("Reply (not sent)", text)
            self.assertIn("Send Like", text)

    def test_prompt_transcript_path_uses_scoop_model_and_time(self):
        when = datetime(2026, 8, 19, 18, 45, 30)
        path = prompt_transcript_path(when, "qwen3.5:9b")
        self.assertEqual(path.name, "Scoop qwen3.5-9b 2026-08-19 18-45-30.txt")
        self.assertEqual(path.parent, Path.home() / "Desktop")

    def test_skip_dismisses_sheet_when_send_like_is_offscreen(self):
        app = self.make_app()
        presses = []
        releases = []
        app.keyboard = SimpleNamespace(
            press=presses.append,
            release=releases.append,
        )
        skip_point = (40, 720)
        capture = SimpleNamespace(
            window=SimpleNamespace(left=0, top=0, width=400, height=800),
            frame=None,
        )
        app.fresh_capture = lambda: capture
        skip_lookups = [None, skip_point]

        def find_skip(_capture):
            point = skip_lookups.pop(0) if skip_lookups else skip_point
            return (None, 0.0) if point is None else (point, 0.92)

        with (
            patch("main_scoop.recognize_text", return_value=[]),
            patch("main_scoop.find_send_priority_like", return_value=(None, 0.0, "")),
            patch("main_scoop.find_hinge_skip_x", side_effect=find_skip),
            patch("main_scoop.viewport_similarity", return_value=0.2),
        ):
            skipped = ScoopUpApp.skip_current_profile(app)

        self.assertTrue(skipped)
        self.assertEqual(presses, [Key.esc])
        self.assertEqual(releases, [Key.esc])
        self.assertEqual(app.clicks, [skip_point])


class OpenAIApiKeyImportTests(unittest.TestCase):
    def test_parse_reads_env_assignment_and_plain_key(self):
        self.assertEqual(
            parse_openai_api_key('export OPENAI_API_KEY="sk-test"\n'),
            "sk-test",
        )
        self.assertEqual(parse_openai_api_key("sk-live-plain\n"), "sk-live-plain")
        self.assertEqual(
            parse_openai_api_key('ANTHROPIC_API_KEY="sk-ant-test"\n'),
            "sk-ant-test",
        )
        self.assertEqual(parse_openai_api_key("# comment\n"), "")


class PaidEngineUiTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
                "GEMINI_API_KEY": "",
                "GOOGLE_API_KEY": "",
                "XAI_API_KEY": "",
                "DEEPSEEK_API_KEY": "",
            },
        )
        self.env.start()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = ScoopUpApp(self.root)
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()
        self.env.stop()

    def test_openai_asks_for_model_then_api_key(self):
        self.assertFalse(self.app.paid_model_menu.winfo_manager())
        self.assertFalse(self.app._api_key_row[1].winfo_manager())

        self.app.engine_var.set(PAID_ENGINE)
        self.root.update_idletasks()

        self.assertTrue(self.app.paid_model_menu.winfo_manager())
        self.assertFalse(self.app._api_key_row[1].winfo_manager())
        self.assertEqual(self.app.paid_model_var.get(), PAID_MODEL_PROMPT)
        self.assertIn("paid model", self.app.status_label.cget("text").lower())
        self.assertIn("Claude Sonnet 4.6", PAY_MODELS)
        self.assertIn("Gemini 2.5 Flash", PAY_MODELS)
        self.assertIn("Grok 4", PAY_MODELS)

        self.app.paid_model_var.set("Claude Sonnet 4.6")
        self.root.update_idletasks()

        self.assertTrue(self.app._api_key_row[1].winfo_manager())
        self.assertIn("anthropic", self.app.status_label.cget("text").lower())

    def test_prompt_reply_shows_save_log_checkbox(self):
        self.assertFalse(self.app._save_transcript_row[1].winfo_manager())
        self.app.workflow_var.set("Prompt Reply")
        self.root.update_idletasks()
        self.assertTrue(self.app._save_transcript_row[1].winfo_manager())
        self.assertFalse(self.app.save_transcript_var.get())


if __name__ == "__main__":
    unittest.main()
