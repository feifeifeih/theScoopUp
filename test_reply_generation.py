import json
import base64
import unittest

from profile_reply import CapturedPrompt
from reply_generation import (
    CloudReplyGenerator,
    GeneratedReply,
    LOCAL_FREE_MODEL,
    OllamaReplyGenerator,
    PAY_MODEL,
    ReplyGenerationError,
    ReplyGenerator,
    TONE_INSTRUCTIONS,
    build_input,
    build_paid_input,
    build_paid_photo_input,
    prepare_reply_for_entry,
    paid_model_from_selection,
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

    def test_paid_input_is_compact_and_uses_only_the_first_prompt(self):
        second = CapturedPrompt(
            "prompt-2",
            "My simple pleasures",
            "Sunday coffee",
            0,
            0,
            (3, 4),
            0.9,
        )

        model_input = build_paid_input(
            [self.prompt, second],
            "Dry & clever",
        )

        self.assertEqual(
            model_input,
            "Generate a Dry & clever reply for this dating profile prompt:\n"
            "Together, we could\n"
            "Dominate pub trivia\n\n"
            "Rules: Return ONLY the reply. No quotes, labels, explanation, or extra text. "
            "Maximum 80 characters. One line only.",
        )
        self.assertNotIn("Sunday coffee", model_input)
        self.assertIn("Rules: Return ONLY the reply", model_input)

    def test_paid_photo_input_includes_house_rules(self):
        text = build_paid_photo_input("Playful & clean")
        self.assertIn("dating profile photo", text)
        self.assertIn("Rules: Return ONLY the reply", text)
        self.assertIn("Maximum 80 characters", text)

    def test_openai_photo_pickup_sends_image_and_house_rules(self):
        image = b"\x89PNG\r\n\x1a\n" + b"x" * 80
        client = FakeClient(["That surfboard looks like trouble."])

        line = ReplyGenerator(client=client).generate_photo_pickup_line(
            image,
            "Playful & clean",
        )

        self.assertIn("surfboard", line.lower())
        call = client.responses.calls[0]
        content = call["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertIn("Rules: Return ONLY the reply", content[0]["text"])
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))

    def test_paid_cloud_photo_providers_send_image_and_house_rules(self):
        image = b"\x89PNG\r\n\x1a\n" + b"x" * 80
        raw_reply = "That orange cat deserves its own fan club."
        anthropic = FakeOpener([{"content": [{"text": raw_reply}]}])
        gemini = FakeOpener([{
            "candidates": [{"content": {"parts": [{"text": raw_reply}]}}],
        }])
        grok = FakeOpener([{
            "choices": [{"message": {"content": raw_reply}}],
        }])

        claude_line = CloudReplyGenerator(
            paid_model_from_selection("Claude Sonnet 4.6"),
            api_key="sk-ant-test",
            opener=anthropic,
        ).generate_photo_pickup_line(image, "Playful & clean")
        flash_line = CloudReplyGenerator(
            paid_model_from_selection("Gemini 2.5 Flash"),
            api_key="gemini-test",
            opener=gemini,
        ).generate_photo_pickup_line(image, "Playful & clean")
        grok_line = CloudReplyGenerator(
            paid_model_from_selection("Grok 4"),
            api_key="xai-test",
            opener=grok,
        ).generate_photo_pickup_line(image, "Flirty & bold")

        self.assertIn("orange cat", claude_line)
        self.assertIn("orange cat", flash_line)
        self.assertIn("orange cat", grok_line)

        anthropic_payload = json.loads(anthropic.requests[0][0].data)
        self.assertEqual(anthropic_payload["messages"][0]["content"][0]["type"], "image")
        self.assertIn("Rules: Return ONLY the reply", anthropic_payload["messages"][0]["content"][1]["text"])

        gemini_payload = json.loads(gemini.requests[0][0].data)
        self.assertIn("inlineData", gemini_payload["contents"][0]["parts"][0])
        self.assertIn("Rules: Return ONLY the reply", gemini_payload["contents"][0]["parts"][1]["text"])

        grok_payload = json.loads(grok.requests[0][0].data)
        self.assertEqual(grok_payload["messages"][0]["content"][1]["type"], "image_url")

    def test_deepseek_photo_fallback_is_unsupported(self):
        image = b"\x89PNG\r\n\x1a\n" + b"x" * 80
        with self.assertRaisesRegex(ReplyGenerationError, "DeepSeek does not support"):
            CloudReplyGenerator(
                paid_model_from_selection("DeepSeek Chat"),
                api_key="deepseek-test",
                opener=FakeOpener([]),
            ).generate_photo_pickup_line(image, "Playful & clean")

    def test_openai_uses_prepared_single_line_text_with_direct_typing(self):
        raw_reply = "  Here's a reply:\nTrivia rivals first 🍟\n"
        client = FakeClient([raw_reply])

        result = ReplyGenerator(client=client).generate(
            [self.prompt],
            "Dry & clever",
        )

        self.assertEqual(result.prompt_id, "prompt-1")
        self.assertEqual(result.reply, "Trivia rivals first")
        call = client.responses.calls[0]
        self.assertEqual(call["model"], PAY_MODEL)
        self.assertEqual(call["input"], build_paid_input([self.prompt], "Dry & clever"))
        self.assertNotIn("text", call)
        self.assertEqual(len(client.responses.calls), 1)

    def test_prepare_reply_for_entry_keeps_best_line_and_ascii(self):
        prepared = prepare_reply_for_entry(
            "Reply:\nPub trivia this serious deserves victory fries 🍟"
        )
        self.assertEqual(prepared, "Pub trivia this serious deserves victory fries")
        self.assertTrue(prepared.isascii())

    def test_uses_selected_pay_model(self):
        client = FakeClient(["A raw reply"])

        ReplyGenerator(client=client, model="gpt-5-mini").generate(
            [self.prompt],
            "Dry & clever",
        )

        self.assertEqual(client.responses.calls[0]["model"], "gpt-5-mini")

    def test_paid_openai_failure_is_not_retried(self):
        client = FakeClient([RuntimeError("offline"), "unused reply"])

        with self.assertRaisesRegex(RuntimeError, "offline"):
            ReplyGenerator(client=client).generate([self.prompt], "Playful & clean")

        self.assertEqual(len(client.responses.calls), 1)

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

    def test_anthropic_and_gemini_models_use_their_own_apis(self):
        raw_reply = "Trivia night sounds like a dangerously fun first plot twist."
        anthropic = FakeOpener([{"content": [{"text": raw_reply}]}])
        gemini = FakeOpener([{
            "candidates": [{"content": {"parts": [{"text": raw_reply}]}}],
        }])

        claude = CloudReplyGenerator(
            paid_model_from_selection("Claude Sonnet 4.6"),
            api_key="sk-ant-test",
            opener=anthropic,
        ).generate([self.prompt], "Playful & clean")
        flash = CloudReplyGenerator(
            paid_model_from_selection("Gemini 2.5 Flash"),
            api_key="gemini-test",
            opener=gemini,
        ).generate([self.prompt], "Playful & clean")

        self.assertEqual(claude.reply, raw_reply)
        self.assertEqual(flash.reply, raw_reply)
        self.assertIn("api.anthropic.com", anthropic.requests[0][0].full_url)
        self.assertIn("generativelanguage.googleapis.com", gemini.requests[0][0].full_url)
        expected_input = build_paid_input([self.prompt], "Playful & clean")
        anthropic_payload = json.loads(anthropic.requests[0][0].data)
        gemini_payload = json.loads(gemini.requests[0][0].data)
        self.assertEqual(
            anthropic_payload["messages"],
            [{"role": "user", "content": expected_input}],
        )
        self.assertNotIn("system", anthropic_payload)
        self.assertEqual(
            gemini_payload["contents"],
            [{"role": "user", "parts": [{"text": expected_input}]}],
        )
        self.assertNotIn("systemInstruction", gemini_payload)
        self.assertNotIn("responseMimeType", gemini_payload["generationConfig"])

    def test_grok_and_deepseek_use_one_raw_user_message(self):
        raw_reply = "Trivia rivalry accepted."
        grok_opener = FakeOpener([
            {"choices": [{"message": {"content": raw_reply}}]},
        ])
        deepseek_opener = FakeOpener([
            {"choices": [{"message": {"content": raw_reply}}]},
        ])

        grok = CloudReplyGenerator(
            paid_model_from_selection("Grok 4"),
            api_key="xai-test",
            opener=grok_opener,
        ).generate([self.prompt], "Flirty & bold")
        deepseek = CloudReplyGenerator(
            paid_model_from_selection("DeepSeek Chat"),
            api_key="deepseek-test",
            opener=deepseek_opener,
        ).generate([self.prompt], "Flirty & bold")

        self.assertEqual(grok.reply, raw_reply)
        self.assertEqual(deepseek.reply, raw_reply)
        expected_messages = [{
            "role": "user",
            "content": build_paid_input([self.prompt], "Flirty & bold"),
        }]
        for opener in (grok_opener, deepseek_opener):
            payload = json.loads(opener.requests[0][0].data)
            self.assertEqual(payload["messages"], expected_messages)
            self.assertEqual(len(opener.requests), 1)

    def test_free_local_generator_uses_ollama_without_an_api_key(self):
        opener = FakeOpener([
            {"models": [{"name": LOCAL_FREE_MODEL}]},
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
        self.assertEqual(payload["model"], LOCAL_FREE_MODEL)
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"]["required"], ["prompt_id", "reply"])

    def test_qwen_photo_generation_sends_image_and_returns_grounded_line(self):
        opener = FakeOpener([
            {"models": [{"name": LOCAL_FREE_MODEL}]},
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
        self.assertEqual(payload["model"], LOCAL_FREE_MODEL)
        self.assertFalse(payload["think"])

    def test_qwen_prompt_detection_sends_viewport_and_extracts_exact_text(self):
        opener = FakeOpener([
            {"models": [{"name": LOCAL_FREE_MODEL}]},
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
        self.assertEqual(payload["model"], LOCAL_FREE_MODEL)
        self.assertFalse(payload["think"])
        self.assertEqual(
            payload["format"]["required"],
            ["has_written_prompt", "prompt", "answer"],
        )

    def test_single_prompt_mode_binds_invented_local_prompt_id(self):
        opener = FakeOpener([
            {"models": [{"name": LOCAL_FREE_MODEL}]},
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
            {"models": [{"name": LOCAL_FREE_MODEL}]},
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
            {"models": [{"name": LOCAL_FREE_MODEL}]},
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
        with self.assertRaises(ReplyGenerationError) as ctx:
            OllamaReplyGenerator(opener=opener).generate(
                [self.prompt],
                "Playful & clean",
            )
        self.assertIn(f"ollama pull {LOCAL_FREE_MODEL}", str(ctx.exception))

    def test_custom_local_free_model_is_requested(self):
        opener = FakeOpener([
            {"models": [{"name": "llama3.2-vision:latest"}]},
            {"response": json.dumps({
                "prompt_id": "prompt-1",
                "reply": "Trivia rivals first, victory fries after?",
            })},
        ])

        result = OllamaReplyGenerator(
            model="llama3.2-vision:latest",
            opener=opener,
        ).generate([self.prompt], "Playful & clean")

        self.assertEqual(result.prompt_id, "prompt-1")
        payload = json.loads(opener.requests[1][0].data)
        self.assertEqual(payload["model"], "llama3.2-vision:latest")


if __name__ == "__main__":
    unittest.main()
