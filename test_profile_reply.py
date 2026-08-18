import unittest
from types import SimpleNamespace

from profile_reply import (
    CapturedPrompt,
    ProfileScanError,
    ProfileScanner,
    ScreenText,
    merge_prompts,
    prompt_is_visible,
    prompts_match,
    prompts_from_viewport,
    recognize_text,
    recover_vision_prompt_target,
    reply_is_visible,
    reply_is_visible_near,
    viewport_signature,
    viewport_similarity,
)


def line(text, top, confidence=0.95, height=20):
    return ScreenText(text, confidence, 80, top, 200, height)


class ProfileParsingTests(unittest.TestCase):
    def test_qwen_prompt_text_is_anchored_to_native_ocr_and_heart(self):
        window = SimpleNamespace(left=0, top=0, width=430, height=800)
        recovered = recover_vision_prompt_target(
            "Together, we could",
            "Win a pub trivia trophy",
            [
                line("Together, we could", 280, height=14),
                line("Win a pub trivia trophy", 310, height=27),
            ],
            [((390, 470), 0.96)],
            1,
            window,
        )

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.prompt, "Together, we could")
        self.assertEqual(recovered.answer, "Win a pub trivia trophy")
        self.assertEqual(recovered.heart_point, (390, 470))

    def test_qwen_prompt_is_rejected_without_a_deterministic_screen_anchor(self):
        window = SimpleNamespace(left=0, top=0, width=430, height=800)
        recovered = recover_vision_prompt_target(
            "Together, we could",
            "Win a pub trivia trophy",
            [line("unrelated profile metadata", 280)],
            [((390, 470), 0.96)],
            0,
            window,
        )

        self.assertIsNone(recovered)

    def test_groups_prompt_and_answer_with_nearest_heart(self):
        prompts = prompts_from_viewport(
            [
                line("Together, we could", 280, height=14),
                line("Win a pub trivia trophy", 310, height=27),
            ],
            [((390, 470), 0.92)],
            viewport_index=2,
            window_height=800,
        )

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].prompt, "Together, we could")
        self.assertEqual(prompts[0].answer, "Win a pub trivia trophy")
        self.assertEqual(prompts[0].scroll_steps, 2)

    def test_long_wording_prompt_heading_is_not_rejected_by_its_center(self):
        prompts = prompts_from_viewport(
            [
                ScreenText(
                    "Something my pet thinks about me",
                    0.95,
                    330,
                    280,
                    300,
                    14,
                ),
                ScreenText(
                    "I torture him by not letting him eat plastic",
                    0.95,
                    330,
                    310,
                    310,
                    28,
                ),
            ],
            [((740, 480), 0.95)],
            viewport_index=0,
            window_height=800,
        )

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].prompt, "Something my pet thinks about me")

    def test_ignores_ui_text_and_low_confidence_ocr(self):
        prompts = prompts_from_viewport(
            [
                line("Send Like", 250),
                line("A life goal of mine", 300, height=14),
                line("Unreadable answer", 330, confidence=0.2, height=27),
            ],
            [((390, 470), 0.95)],
            viewport_index=0,
            window_height=800,
        )
        self.assertEqual(prompts, [])

    def test_profile_thoughtful_signals_banner_is_not_a_prompt(self):
        for banner in (
            "Sophia shows thoughtful signals",
            "Amy shows thoughtful signal",
            "Caitlin shows thoughtful signals • Verified",
        ):
            with self.subTest(banner=banner):
                prompts = prompts_from_viewport(
                    [
                        line(banner, 280, height=14),
                        line("Verified signals", 310, height=27),
                    ],
                    [((390, 470), 0.95)],
                    viewport_index=0,
                    window_height=800,
                )
                self.assertEqual(prompts, [])

    def test_profile_filter_bar_labels_are_not_prompts(self):
        for label_text in ("Age V", "Height V", "Dating Intent", "Signals"):
            with self.subTest(label=label_text):
                prompts = prompts_from_viewport(
                    [
                        line(label_text, 280, height=14),
                        line("Some adjacent UI text", 310, height=27),
                    ],
                    [((390, 470), 0.95)],
                    viewport_index=0,
                    window_height=800,
                )
                self.assertEqual(prompts, [])

    def test_does_not_attach_prompt_below_a_photo_heart(self):
        prompts = prompts_from_viewport(
            [
                line("My simple pleasures", 650, height=14),
                line("Coffee and coloring", 680, height=27),
            ],
            [((390, 560), 0.95)],
            viewport_index=0,
            window_height=800,
        )
        self.assertEqual(prompts, [])

    def test_does_not_associate_metadata_with_clipped_bottom_heart(self):
        prompts = prompts_from_viewport(
            [
                line("White/Caucasian", 370, height=12),
                line("Life partner", 413, height=14),
                line("Monogamy", 459, height=13),
            ],
            [((390, 780), 0.95)],
            viewport_index=0,
            window_height=781,
            window_top=40,
        )

        self.assertEqual(prompts, [])

    def test_does_not_treat_single_word_profile_metadata_as_a_prompt(self):
        prompts = prompts_from_viewport(
            [line("Liberal", 280, height=14), line("Long-term relationship", 310, height=27)],
            [((390, 470), 0.95)],
            viewport_index=0,
            window_height=800,
        )
        self.assertEqual(prompts, [])

    def test_does_not_treat_status_bar_time_as_a_prompt(self):
        prompts = prompts_from_viewport(
            [line("23:32 (", 280, height=14), line("Abigail", 310, height=27)],
            [((390, 470), 0.95)],
            viewport_index=0,
            window_height=800,
        )
        self.assertEqual(prompts, [])

    def test_does_not_treat_new_here_badge_or_answer_fragment_as_prompt(self):
        for heading in ("New here", "Micro-aggressions. Colorism. Caste"):
            prompts = prompts_from_viewport(
                [line(heading, 280, height=14), line("Verified profile", 310, height=27)],
                [((390, 470), 0.95)],
                viewport_index=0,
                window_height=800,
            )
            self.assertEqual(prompts, [])

    def test_ignores_lets_get_together_card(self):
        prompts = prompts_from_viewport(
            [
                line("Let's get together", 280, height=14),
                line("Go pottery painting", 310, height=27),
            ],
            [((390, 470), 0.95)],
            viewport_index=0,
            window_height=800,
        )
        self.assertEqual(prompts, [])

    def test_lowercase_wrapped_answer_line_does_not_replace_heading(self):
        prompts = prompts_from_viewport(
            [
                line("What if I told you that", 280, height=14),
                line("I am a background cast", 310, height=24),
                line("member in the new", 334, height=21),
                line("Christopher Nolan", 360, height=23),
                line("adaptation of the", 384, height=20),
                line("odyssey", 407, height=25),
            ],
            [((390, 470), 0.95)],
            viewport_index=0,
            window_height=800,
        )

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].prompt, "What if I told you that")

    def test_overlapping_similar_prompts_are_deduplicated(self):
        first = CapturedPrompt("one", "My simple pleasures", "Strong coffee", 0, 0, (1, 2), 0.8)
        second = CapturedPrompt("two", "My simple pleasure", "Strong coffee!", 1, 1, (3, 4), 0.9)

        merged = merge_prompts([first], [second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], second)

    def test_relocation_matching_tolerates_changed_long_answer_ocr(self):
        target = CapturedPrompt(
            "old", "The way to win me over is",
            "Plan a thoughtful date and hold my hand through awkward small talk",
            0, 0, (1, 2), 0.9,
        )
        fresh = CapturedPrompt(
            "new", "The way to win me over is", "Plan a thoughtful date",
            1, 1, (3, 4), 0.9,
        )
        self.assertTrue(prompts_match(fresh, target))

    def test_lowercase_answer_fragment_is_not_a_prompt_heading(self):
        prompts = prompts_from_viewport(
            [
                line("excited to spend time", 280, height=14),
                line("together", 310, height=27),
            ],
            [((390, 470), 0.95)],
            viewport_index=0,
            window_height=800,
        )
        self.assertEqual(prompts, [])

    def test_reply_visibility_handles_wrapped_ocr(self):
        lines = [line("Finally, a worthy", 200), line("trivia rival appears", 240)]
        self.assertTrue(reply_is_visible(lines, "Finally, a worthy trivia rival appears"))

    def test_reply_visibility_accepts_visible_composer_prefix(self):
        lines = [line("Finally, a w", 200)]
        self.assertTrue(
            reply_is_visible(lines, "Finally, a worthy trivia rival appears")
        )

    def test_reply_visibility_accepts_visible_composer_suffix(self):
        lines = [line("coffee snob. I'm here to help you", 200)]
        self.assertTrue(
            reply_is_visible(
                lines,
                "Sounds like becoming the ultimate coffee snob. I'm here to help you",
            )
        )

    def test_open_dialog_must_match_selected_prompt_and_answer(self):
        selected = CapturedPrompt(
            "travel",
            "Best travel story",
            "My passport expired before Bali",
            0,
            0,
            (1, 2),
            0.9,
        )
        travel_lines = [
            line("Best travel story", 200),
            line("My passport expired before Bali", 240),
        ]
        sushi_lines = [
            line("Together, we could", 200),
            line("Eat sushi on a first date", 240),
        ]

        self.assertTrue(prompt_is_visible(travel_lines, selected))
        self.assertFalse(prompt_is_visible(sushi_lines, selected))

    def test_reply_verification_ignores_matching_words_outside_composer(self):
        lines = [
            line("Harry Potter and break out in song", 300),
            line("Add a comment", 600),
        ]
        self.assertFalse(
            reply_is_visible_near(
                lines,
                "Harry Potter fans should break out in song",
                (200, 610),
                800,
            )
        )

    def test_visual_signature_detects_movement_in_text_free_photos(self):
        class Frame:
            width = 100
            height = 200

            def __init__(self, color):
                self.color = color

            def color_at(self, _x, _y):
                return self.color

        self.assertNotEqual(
            viewport_signature([], Frame((10, 20, 30))),
            viewport_signature([], Frame((210, 220, 230))),
        )

    def test_viewport_similarity_tolerates_minor_ocr_jitter(self):
        before = [line("My simple pleasures", 200), line("Coffee and coloring", 250)]
        after = [line("My simple pleasures", 201), line("Coffee & coloring", 251)]
        self.assertGreater(viewport_similarity(before, None, after, None), 0.94)


class ProfileScannerTests(unittest.TestCase):
    def setUp(self):
        self.position = 2
        self.views = [
            [line("Together, we could", 280, height=14), line("Win trivia night", 310, height=27)],
            [line("My simple pleasures", 280, height=14), line("Coffee before talking", 310, height=27)],
            [line("The key to my heart", 280, height=14), line("Excellent dumplings", 310, height=27)],
        ]
        self.capture_object = SimpleNamespace(window=SimpleNamespace(height=800))

    def capture(self):
        return self.capture_object

    def ocr(self, _capture):
        return self.views[self.position]

    def hearts(self, _capture):
        return [((390, 470), 0.95)]

    def scroll(self, direction):
        if direction == "up":
            self.position = max(0, self.position - 1)
        else:
            self.position = min(len(self.views) - 1, self.position + 1)

    def scanner(self, prompt_limit=1):
        return ProfileScanner(
            self.capture,
            self.ocr,
            self.hearts,
            self.scroll,
            lambda _seconds: None,
            max_viewports=12,
            prompt_limit=prompt_limit,
        )

    def test_stops_scanning_as_soon_as_first_written_prompt_is_found(self):
        scan = self.scanner().scan()

        self.assertEqual(len(scan.prompts), 1)
        self.assertEqual(scan.viewport_count, 1)
        self.assertEqual(self.position, 0)

    def test_fast_jump_skips_the_blank_profile_header(self):
        self.position = 0
        self.views[0] = [line("Taylor", 280, height=24)]
        scrolls = []

        def scroll(direction):
            scrolls.append(direction)
            self.scroll(direction)

        scanner = ProfileScanner(
            self.capture,
            self.ocr,
            self.hearts,
            scroll,
            lambda _seconds: None,
            fast_jump_viewports=1,
        )

        scan = scanner.scan(ensure_top=False)

        self.assertEqual(scan.viewport_count, 1)
        self.assertEqual(scan.prompts[0].prompt, "My simple pleasures")
        self.assertEqual(scrolls, ["down"])

    def test_local_vision_rescue_runs_only_once_per_scan(self):
        self.position = 0
        self.views = [[line("Taylor", 280)], [line("Designer", 280)]]
        calls = []

        def rescue(_capture, _lines, _hearts, viewport_index):
            calls.append(viewport_index)
            return None

        scanner = ProfileScanner(
            self.capture,
            self.ocr,
            self.hearts,
            self.scroll,
            lambda _seconds: None,
            max_viewports=2,
            vision_rescue=rescue,
            vision_rescue_delay=0,
        )

        with self.assertRaises(ProfileScanError):
            scanner.scan(ensure_top=False)

        self.assertEqual(calls, [0])

    def test_local_vision_rescue_waits_five_seconds_without_an_ocr_prompt(self):
        self.position = 0
        self.views = [[line("Taylor", 280)]]
        now = [0.0]
        waits = []
        calls = []
        rescued_prompt = CapturedPrompt(
            "vision-prompt",
            "Together, we could",
            "Win trivia night",
            0,
            0,
            (390, 470),
            0.8,
        )

        def wait(seconds):
            waits.append(seconds)
            now[0] += seconds

        def rescue(_capture, _lines, _hearts, viewport_index):
            calls.append((viewport_index, now[0]))
            return rescued_prompt

        scanner = ProfileScanner(
            self.capture,
            self.ocr,
            self.hearts,
            self.scroll,
            wait,
            max_viewports=1,
            vision_rescue=rescue,
            vision_rescue_delay=5,
            clock=lambda: now[0],
        )

        scan = scanner.scan(ensure_top=False)

        self.assertEqual(scan.prompts, (rescued_prompt,))
        self.assertEqual(waits, [5.0])
        self.assertEqual(calls, [(1, 5.0)])

    def test_continues_past_ignored_card_to_first_written_prompt(self):
        self.views[0] = [
            line("Let's get together for", 280, height=14),
            line("Go pottery painting", 310, height=27),
        ]
        self.views = self.views[:2]
        self.position = 1

        scan = self.scanner().scan()

        self.assertEqual(len(scan.prompts), 1)
        self.assertEqual(scan.prompts[0].prompt, "My simple pleasures")
        self.assertEqual(scan.viewport_count, 2)
        self.assertEqual(self.position, 1)

    def test_relocates_selected_prompt_from_fresh_capture(self):
        scan = self.scanner(prompt_limit=3).scan()
        target = scan.prompts[1]

        relocated = self.scanner().relocate(target)

        self.assertEqual(relocated.prompt_id, target.prompt_id)
        self.assertEqual(self.position, 1)

    def test_reconfirms_first_prompt_without_scrolling_to_top(self):
        scan = self.scanner().scan()
        target = scan.prompts[0]
        position_before = self.position

        reconfirmed = self.scanner().reconfirm_visible(target)

        self.assertEqual(reconfirmed.prompt_id, target.prompt_id)
        self.assertEqual(self.position, position_before)

    def test_relocation_searches_beyond_stale_recorded_scroll_position(self):
        scan = self.scanner(prompt_limit=3).scan()
        original = scan.prompts[2]
        stale_target = CapturedPrompt(
            original.prompt_id,
            original.prompt,
            original.answer,
            original.viewport_index,
            0,
            original.heart_point,
            original.confidence,
        )

        relocated = self.scanner(prompt_limit=3).relocate(stale_target)

        self.assertEqual(relocated.prompt_id, original.prompt_id)
        self.assertEqual(self.position, 2)

    def test_selected_prompt_is_centered_before_clicking(self):
        offset = 0
        capture = SimpleNamespace(window=SimpleNamespace(top=0, height=800))
        capture_calls = 0

        def capture_current():
            nonlocal capture_calls
            capture_calls += 1
            return capture

        def lines():
            shift = offset * 100
            return [
                line("My simple pleasures", 510 - shift, height=14),
                line("Coffee before talking", 540 - shift, height=27),
            ]

        def hearts(_capture):
            return [((390, 700 - offset * 100), 0.95)]

        def scroll(direction):
            nonlocal offset
            self.assertIn(direction, ("down", "down_small"))
            offset += 1

        scanner = ProfileScanner(
            capture_current,
            lambda _capture: lines(),
            hearts,
            scroll,
            lambda _seconds: None,
        )
        initial = prompts_from_viewport(lines(), hearts(capture), 0, 800)[0]

        centered = scanner.center_target(initial, initial)

        self.assertEqual(centered.heart_point[1], 500)
        self.assertEqual(offset, 2)
        self.assertEqual(capture_calls, 3)

    def test_centering_uses_only_small_nudges_to_keep_prompt_visible(self):
        capture = SimpleNamespace(window=SimpleNamespace(top=0, height=800))
        offset = 0
        directions = []

        def lines():
            return [
                line("My simple pleasures", 610 - offset * 100, height=14),
                line("Coffee before talking", 640 - offset * 100, height=27),
            ]

        def hearts(_capture):
            return [((390, 700 - offset * 100), 0.95)]

        def scroll(direction):
            nonlocal offset
            directions.append(direction)
            offset += 1

        scanner = ProfileScanner(
            lambda: capture,
            lambda _capture: lines(),
            hearts,
            scroll,
            lambda _seconds: None,
        )
        initial = prompts_from_viewport(lines(), hearts(capture), 0, 800)[0]

        scanner.center_target(initial, initial)

        self.assertTrue(directions)
        self.assertEqual(set(directions), {"down_small"})

    def test_already_centered_prompt_is_not_scrolled(self):
        capture = SimpleNamespace(window=SimpleNamespace(top=0, height=800))
        lines = [
            line("My simple pleasures", 300, height=14),
            line("Coffee before talking", 340, height=27),
        ]
        hearts = [((390, 460), 0.95)]
        scrolls = []
        scanner = ProfileScanner(
            lambda: capture,
            lambda _capture: lines,
            lambda _capture: hearts,
            scrolls.append,
            lambda _seconds: None,
        )
        initial = prompts_from_viewport(lines, hearts, 0, 800)[0]

        centered = scanner.center_target(initial, initial)

        self.assertEqual(centered, initial)
        self.assertEqual(scrolls, [])

    def test_prompt_in_top_thirty_percent_is_not_scrolled(self):
        capture = SimpleNamespace(window=SimpleNamespace(top=0, height=800))
        lines = [
            line("My simple pleasures", 100, height=14),
            line("Coffee before talking", 140, height=27),
        ]
        hearts = [((390, 240), 0.95)]
        scrolls = []
        scanner = ProfileScanner(
            lambda: capture,
            lambda _capture: lines,
            lambda _capture: hearts,
            scrolls.append,
            lambda _seconds: None,
        )
        initial = prompts_from_viewport(lines, hearts, 0, 800)[0]

        centered = scanner.center_target(initial, initial)

        self.assertEqual(centered, initial)
        self.assertEqual(scrolls, [])

    def test_safe_heart_is_not_scrolled_when_heading_ocr_temporarily_misses(self):
        capture = SimpleNamespace(window=SimpleNamespace(top=0, height=800))
        initial_lines = [
            line("My simple pleasures", 100, height=14),
            line("Coffee before talking", 140, height=27),
        ]
        hearts = [((390, 240), 0.95)]
        initial = prompts_from_viewport(initial_lines, hearts, 0, 800)[0]
        scrolls = []
        scanner = ProfileScanner(
            lambda: capture,
            lambda _capture: [],
            lambda _capture: hearts,
            scrolls.append,
            lambda _seconds: None,
        )

        positioned = scanner.center_target(initial, initial)

        self.assertEqual(positioned, initial)
        self.assertEqual(scrolls, [])

    def test_fast_ocr_disables_language_correction(self):
        levels = []
        corrections = []

        class FakeRequest:
            def setRecognitionLevel_(self, level):
                levels.append(level)

            def setUsesLanguageCorrection_(self, value):
                corrections.append(value)

            def results(self):
                return []

        class FakeRequestType:
            def alloc(self):
                return self

            def init(self):
                return FakeRequest()

        class FakeHandler:
            def alloc(self):
                return self

            def initWithCGImage_options_(self, _image, _options):
                return self

            def performRequests_error_(self, _requests, _error):
                return True, None

        vision = SimpleNamespace(
            VNRecognizeTextRequest=FakeRequestType(),
            VNRequestTextRecognitionLevelAccurate="accurate",
            VNRequestTextRecognitionLevelFast="fast",
            VNImageRequestHandler=FakeHandler(),
        )
        capture = SimpleNamespace(
            frame=SimpleNamespace(width=10, height=10),
            window=SimpleNamespace(width=10, height=10),
            ensure_vision_image=lambda: object(),
        )

        recognize_text(capture, vision, accurate=False)
        recognize_text(capture, vision, accurate=True)

        self.assertEqual(levels, ["fast", "accurate"])
        self.assertEqual(corrections, [False, True])


if __name__ == "__main__":
    unittest.main()
