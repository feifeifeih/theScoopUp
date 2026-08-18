import json
import base64
import unittest

from profile_reply import CapturedPrompt
from reply_generation import (
    GeneratedReply,
    OllamaReplyGenerator,
    ReplyGenerationError,
    ReplyGenerator,
    TONE_INSTRUCTIONS,
    build_input,
    random_pickup_line,
    validate_fallback_pickup_line,
    validate_photo_pickup_line,
    validate_grounding,
    validate_reply,
)


class FakeResponses:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return type("Response", (), {"output_text": payload})()


class FallbackPickupLineTests(unittest.TestCase):
    def test_random_pickup_line_uses_injected_choice_and_stays_sendable(self):
        chosen = "Coffee or tacos: which one earns the first-date trophy?"
        line = random_pickup_line(lambda _options: chosen)
        self.assertEqual(line, chosen)
        self.assertLessEqual(len(line), 140)
        self.assertNotIn("\n", line)

    def test_custom_pickup_line_rejects_multiline_and_unsafe_text(self):
        with self.assertRaises(ReplyGenerationError):
            validate_fallback_pickup_line("Hello\nthere")
        with self.assertRaises(ReplyGenerationError):
            validate_fallback_pickup_line("Send nudes")

    def test_custom_pickup_line_accepts_clean_text(self):
        line = "Coffee first, or should we jump straight to planning an adventure?"
        self.assertEqual(validate_fallback_pickup_line(line), line)

    def test_photo_line_must_be_grounded_and_avoid_appearance(self):
        self.assertEqual(
            validate_photo_pickup_line(
                "Does that surfboard come with lessons, or just confidence?",
                "a surfboard by the ocean",
            ),
            "Does that surfboard come with lessons, or just confidence?",
        )
        with self.assertRaisesRegex(ReplyGenerationError, "concrete detail"):
            validate_photo_pickup_line(
                "Coffee first, or should we plan an adventure?",
                "a surfboard by the ocean",
            )
        with self.assertRaisesRegex(ReplyGenerationError, "unsafe visual"):
            validate_photo_pickup_line(
                "Your gorgeous smile deserves a coffee",
                "a gorgeous smile",
            )
        with self.assertRaisesRegex(ReplyGenerationError, "compared"):
            validate_photo_pickup_line(
                "You're my favorite orange cat to stare at.",
                "orange cat",
            )


class FakeClient:
    def __init__(self, payloads):
        self.responses = FakeResponses(payloads)


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return FakeHTTPResponse(self.payloads.pop(0))


class ReplyGenerationTests(unittest.TestCase):
    def setUp(self):
        self.prompt = CapturedPrompt(
            "prompt-1",
            "Together, we could",
            "Dominate pub trivia",
            0,
            0,
            (1, 2),
            0.9,
        )

    def test_every_tone_is_added_to_the_system_prompt(self):
        for tone, instruction in TONE_INSTRUCTIONS.items():
            messages = build_input([self.prompt], tone)
            self.assertIn(instruction, messages[0]["content"])

    def test_generation_must_reply_to_the_answer_not_only_the_heading(self):
        messages = build_input([self.prompt], "Playful & clean")
        self.assertIn("concrete detail", messages[0]["content"])
        self.assertIn("do not respond to the prompt heading alone", messages[0]["content"])
        self.assertIn("reply directly to that answer", messages[1]["content"])

    def test_uses_responses_structured_output(self):
        client = FakeClient([json.dumps({
            "prompt_id": "prompt-1",
            "reply": "I bring obscure trivia facts and dangerously confident guesses.",
        })])

        result = ReplyGenerator(client=client).generate([self.prompt], "Dry & clever")

        self.assertEqual(result.prompt_id, "prompt-1")
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-5.6-luna")
        self.assertTrue(call["text"]["format"]["strict"])

    def test_retries_once_after_malformed_output(self):
        client = FakeClient([
            "not json",
            json.dumps({
                "prompt_id": "prompt-1",
                "reply": "Trivia rivals first, celebratory fries second?",
            }),
        ])

        result = ReplyGenerator(client=client).generate([self.prompt], "Playful & clean")

        self.assertEqual(result.reply, "Trivia rivals first, celebratory fries second?")
        self.assertEqual(len(client.responses.calls), 2)

    def test_rejects_unknown_prompt_and_unsafe_or_generic_text(self):
        with self.assertRaises(ReplyGenerationError):
            validate_reply(GeneratedReply("unknown", "A specific funny response"), ["prompt-1"])
        with self.assertRaises(ReplyGenerationError):
            validate_reply(GeneratedReply("prompt-1", "Tell me more"), ["prompt-1"])
        with self.assertRaises(ReplyGenerationError):
            validate_reply(GeneratedReply("prompt-1", "That idea is stupid and ugly"), ["prompt-1"])
        with self.assertRaises(ReplyGenerationError):
            validate_reply(
                GeneratedReply(
                    "prompt-1",
                    "Your diva-sized ego is on the verge of collapse",
                ),
                ["prompt-1"],
            )
        with self.assertRaisesRegex(ReplyGenerationError, "truncated"):
            validate_reply(
                GeneratedReply("prompt-1", "A specific reply that stops midway,"),
                ["prompt-1"],
            )
        with self.assertRaisesRegex(ReplyGenerationError, "unmatched quote"):
            validate_reply(
                GeneratedReply("prompt-1", "A specific reply ends oddly '"),
                ["prompt-1"],
            )

    def test_rejects_reply_without_a_concrete_prompt_word(self):
        with self.assertRaisesRegex(ReplyGenerationError, "concrete word"):
            validate_grounding(
                GeneratedReply("prompt-1", "Guess that's one way to show you care"),
                [self.prompt],
            )
        grounded = validate_grounding(
            GeneratedReply("prompt-1", "Trivia this serious deserves a trophy"),
            [self.prompt],
        )
        self.assertIn("Trivia", grounded.reply)

    def test_non_ascii_decoration_is_removed_for_reliable_direct_typing(self):
        reply = validate_reply(
            GeneratedReply("prompt-1", "Trivia rivals deserve fries! 🍟"),
            ["prompt-1"],
        )

        self.assertEqual(reply.reply, "Trivia rivals deserve fries!")
        self.assertTrue(reply.reply.isascii())

    def test_two_failures_are_fail_closed(self):
        client = FakeClient([RuntimeError("offline"), RuntimeError("still offline")])
        with self.assertRaises(ReplyGenerationError):
            ReplyGenerator(client=client).generate([self.prompt], "Playful & clean")
        self.assertEqual(len(client.responses.calls), 2)

    def test_free_local_generator_uses_ollama_without_an_api_key(self):
        opener = FakeOpener([
            {"models": [{"name": "qwen3.5:9b"}]},
            {"response": json.dumps({
                "prompt_id": "prompt-1",
                "reply": "Trivia rivals first, victory fries after?",
            })},
        ])

        result = OllamaReplyGenerator(opener=opener).generate(
            [self.prompt],
            "Playful & clean",
        )

        self.assertEqual(result.prompt_id, "prompt-1")
        generation_request = opener.requests[1][0]
        payload = json.loads(generation_request.data)
        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"]["required"], ["prompt_id", "reply"])

    def test_qwen_photo_generation_sends_image_and_returns_grounded_line(self):
        opener = FakeOpener([
            {"models": [{"name": "qwen3.5:9b"}]},
            {"message": {"content": json.dumps({
                "usable": True,
                "category": "object",
                "visual_detail": "a surfboard by the ocean",
                "line": "Does that surfboard come with lessons, or just confidence?",
            })}},
        ])
        image = b"\x89PNG\r\n\x1a\n" + b"x" * 80

        line = OllamaReplyGenerator(opener=opener).generate_photo_pickup_line(
            image,
            "Playful & clean",
        )

        self.assertIn("surfboard", line)
        request = opener.requests[1][0]
        self.assertTrue(request.full_url.endswith("/api/chat"))
        payload = json.loads(request.data)
        encoded = payload["messages"][0]["images"][0]
        self.assertEqual(base64.b64decode(encoded), image)
        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertFalse(payload["think"])

    def test_qwen_prompt_detection_sends_viewport_and_extracts_exact_text(self):
        opener = FakeOpener([
            {"models": [{"name": "qwen3.5:9b"}]},
            {"message": {"content": json.dumps({
                "has_written_prompt": True,
                "prompt": "Together, we could",
                "answer": "Win a pub trivia trophy",
            })}},
        ])
        image = b"\x89PNG\r\n\x1a\n" + b"x" * 80

        detected = OllamaReplyGenerator(opener=opener).detect_written_prompt(image)

        self.assertEqual(
            detected,
            ("Together, we could", "Win a pub trivia trophy"),
        )
        request = opener.requests[1][0]
        payload = json.loads(request.data)
        self.assertEqual(base64.b64decode(payload["messages"][0]["images"][0]), image)
        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertFalse(payload["think"])
        self.assertEqual(
            payload["format"]["required"],
            ["has_written_prompt", "prompt", "answer"],
        )

    def test_single_prompt_mode_binds_invented_local_prompt_id(self):
        opener = FakeOpener([
            {"models": [{"name": "qwen3.5:9b"}]},
            {"response": json.dumps({
                "prompt_id": "invented-id",
                "reply": "Trivia rivals deserve dramatically oversized trophies.",
            })},
        ])

        result = OllamaReplyGenerator(opener=opener).generate(
            [self.prompt],
            "Playful & clean",
        )

        self.assertEqual(result.prompt_id, "prompt-1")

    def test_local_grounding_retry_locks_to_selected_prompt(self):
        opener = FakeOpener([
            {"models": [{"name": "qwen3.5:9b"}]},
            {"response": json.dumps({
                "prompt_id": "prompt-1",
                "reply": "Guess that's one way to show you care",
            })},
            {"response": json.dumps({
                "prompt_id": "prompt-1",
                "reply": "Pub trivia this serious deserves victory fries",
            })},
        ])

        result = OllamaReplyGenerator(opener=opener).generate(
            [self.prompt],
            "Playful & clean",
        )

        self.assertTrue(
            any(word in result.reply.casefold() for word in ("dominate", "trivia"))
        )
        retry_payload = json.loads(opener.requests[2][0].data)
        self.assertIn("required_content_words", retry_payload["prompt"])

    def test_local_grounding_failures_use_safe_single_prompt_fallback(self):
        ungrounded = {
            "response": json.dumps({
                "prompt_id": "prompt-1",
                "reply": "Guess that's one way to show you care",
            })
        }
        opener = FakeOpener([
            {"models": [{"name": "qwen3.5:9b"}]},
            ungrounded,
            ungrounded,
            ungrounded,
            ungrounded,
        ])

        result = OllamaReplyGenerator(opener=opener).generate(
            [self.prompt],
            "Playful & clean",
        )

        self.assertEqual(result.prompt_id, "prompt-1")
        self.assertTrue(
            any(word in result.reply.casefold() for word in ("dominate", "trivia"))
        )

    def test_free_local_generator_reports_missing_model(self):
        opener = FakeOpener([{"models": []}])
        with self.assertRaisesRegex(ReplyGenerationError, "ollama pull qwen3.5:9b"):
            OllamaReplyGenerator(opener=opener).generate(
                [self.prompt],
                "Playful & clean",
            )


if __name__ == "__main__":
    unittest.main()
