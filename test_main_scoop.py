import unittest

from main_scoop import (
    HeartIconDetector,
    HINGE_HEART_BUTTON_TEMPLATE,
    MirroringWindow,
    PixelFrame,
    SCREEN_RECORDING_SETTINGS_URL,
    ScoopUpApp,
    WindowCapture,
    _normalize_text,
    _window_score,
)


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

    def test_plain_black_circle_is_not_the_hinge_heart_button(self):
        size = 72
        circle = [
            (x, y)
            for y in range(size)
            for x in range(size)
            if (x - 35.5) ** 2 + (y - 35.5) ** 2 <= 35 ** 2
        ]
        self.assertLess(
            self.detector.hinge_button_score(circle, (0, 0, size, size)),
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
        self.assertEqual(_normalize_text("Send Priority Like!"), "send priority like")

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


if __name__ == "__main__":
    unittest.main()
