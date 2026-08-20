"""Paid raw-text and local validated funny reply generation."""

from dataclasses import dataclass
import base64
import json
import os
import random
import re
import shutil
import subprocess
import time
from urllib import error as url_error
from urllib import request as url_request


@dataclass(frozen=True)
class PaidModel:
    label: str
    provider: str
    model_id: str
    env_key: str

    @property
    def provider_title(self):
        return {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "google": "Google",
            "xai": "xAI",
            "deepseek": "DeepSeek",
        }[self.provider]


PAID_MODELS = (
    PaidModel("OpenAI gpt-5.6-luna", "openai", "gpt-5.6-luna", "OPENAI_API_KEY"),
    PaidModel("OpenAI gpt-5", "openai", "gpt-5", "OPENAI_API_KEY"),
    PaidModel("OpenAI gpt-5-mini", "openai", "gpt-5-mini", "OPENAI_API_KEY"),
    PaidModel("OpenAI gpt-5-nano", "openai", "gpt-5-nano", "OPENAI_API_KEY"),
    PaidModel("Claude Sonnet 4.6", "anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    PaidModel("Claude Haiku 4.5", "anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    PaidModel("Gemini 2.5 Flash", "google", "gemini-2.5-flash", "GEMINI_API_KEY"),
    PaidModel("Gemini 2.5 Pro", "google", "gemini-2.5-pro", "GEMINI_API_KEY"),
    PaidModel("Grok 4", "xai", "grok-4", "XAI_API_KEY"),
    PaidModel("DeepSeek Chat", "deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),
)
PAY_MODELS = tuple(model.label for model in PAID_MODELS)
PAY_MODEL = PAID_MODELS[0].model_id
PAID_ENGINE = "Paid API"
API_KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
)
LOCAL_FREE_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MAX_REPLY_LENGTH = 140
PAID_MAX_REPLY_LENGTH = 80
PAID_HOUSE_RULES = (
    "Rules: Return ONLY the reply. No quotes, labels, explanation, or extra text. "
    f"Maximum {PAID_MAX_REPLY_LENGTH} characters. One line only."
)
TONE_INSTRUCTIONS = {
    "Playful & clean": "Warm, playful, clever, and clean. No innuendo or appearance jokes.",
    "Flirty & bold": "Confident and flirty with light teasing, but never explicit or insulting.",
    "Dry & clever": "Concise, understated, and witty; prefer dry wordplay over overt flirting.",
}
FALLBACK_PICKUP_LINES = (
    "Quick question: are we calling this a meet-cute now or after coffee?",
    "You seem worth a bold hello, so: coffee, tacos, or a wildly competitive walk?",
    "I had a clever opener rehearsed, but your profile made me forget my lines.",
    "Let's skip the tiny talk: what's your most defensible unpopular opinion?",
    "On a scale from coffee to cocktails, how spontaneous is our first date?",
    "I'm taking a chance on your vibe. What's the best first-date plot twist?",
)


class ReplyGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedReply:
    prompt_id: str
    reply: str


def random_pickup_line(chooser=None):
    """Choose a locally stored, clean fallback opener without an API call."""
    chooser = chooser or random.SystemRandom().choice
    return validate_fallback_pickup_line(chooser(FALLBACK_PICKUP_LINES))


GENERIC_REPLIES = {
    "hey",
    "hi",
    "hello",
    "thats funny",
    "that is funny",
    "i love this",
    "nice",
    "sounds fun",
    "tell me more",
}
UNSAFE_PATTERNS = (
    r"\b(?:idiot|stupid|ugly|fat|loser|crazy)\b",
    r"\b(?:diva[- ]sized ego|ego on the verge of collapse)\b",
    r"\b(?:nudes?|naked|sex|horny)\b",
    r"https?://|www\.",
    r"@[a-z0-9_]{2,}",
)
PHOTO_UNSAFE_PATTERNS = UNSAFE_PATTERNS + (
    r"\b(?:body|skin|ethnicity|race|religion|disability|disabled|weight)\b",
    r"\b(?:young|old|thin|curvy|muscular|handsome|beautiful|gorgeous|hot|sexy)\b",
)
GROUNDING_STOPWORDS = {
    "about", "after", "again", "always", "could", "first", "from", "going",
    "have", "into", "just", "like", "mine", "more", "really", "something",
    "that", "their", "them", "then", "there", "these", "they", "thing",
    "thinks", "this", "those", "together", "very", "what", "when", "where",
    "which", "with", "would", "your",
}
PHOTO_PICKUP_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "category": {
            "type": "string",
            "enum": ["pet", "activity", "food", "landmark", "object", "setting"],
        },
        "visual_detail": {"type": "string", "maxLength": 60},
        "line": {"type": "string", "maxLength": MAX_REPLY_LENGTH},
    },
    "required": ["usable", "category", "visual_detail", "line"],
    "additionalProperties": False,
}
PROMPT_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "has_written_prompt": {"type": "boolean"},
        "prompt": {"type": "string", "maxLength": 160},
        "answer": {"type": "string", "maxLength": 500},
    },
    "required": ["has_written_prompt", "prompt", "answer"],
    "additionalProperties": False,
}


def validate_fallback_pickup_line(line):
    """Validate a fallback pickup line before it can reach Hinge."""
    if not isinstance(line, str):
        raise ReplyGenerationError("The fallback pickup line must be text.")
    text = line.strip()
    if not text or "\n" in text or "\r" in text:
        raise ReplyGenerationError("The fallback pickup line must be one non-empty line.")
    if len(text) > MAX_REPLY_LENGTH:
        raise ReplyGenerationError(
            f"The fallback pickup line exceeds {MAX_REPLY_LENGTH} characters."
        )
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
        raise ReplyGenerationError("The fallback pickup line failed local safety validation.")
    return text


def validate_photo_pickup_line(line, visual_detail):
    """Require a safe line to repeat a concrete detail reported from the image."""
    text = validate_fallback_pickup_line(line)
    text = text.encode("ascii", "ignore").decode("ascii").strip()
    detail = str(visual_detail).encode("ascii", "ignore").decode("ascii").strip()
    if not detail or len(detail) > 60:
        raise ReplyGenerationError("The photo description was missing or too long.")
    if any(
        re.search(pattern, f"{detail} {text}", re.IGNORECASE)
        for pattern in PHOTO_UNSAFE_PATTERNS
    ):
        raise ReplyGenerationError("The photo pickup line used an unsafe visual attribute.")

    def concrete_words(value):
        return {
            word[:-1] if word.endswith("s") and len(word) > 4 else word
            for word in re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
            if len(word) >= 3 and word not in GROUNDING_STOPWORDS
        }

    details = concrete_words(detail)
    if not details or not details.intersection(concrete_words(text)):
        raise ReplyGenerationError(
            "The pickup line did not repeat a concrete detail from the photo description."
        )
    if re.search(
        r"\byou(?:'re| are)\b[^.!?]{0,30}\b(?:cat|dog|puppy|kitten|animal|object)\b",
        text,
        re.IGNORECASE,
    ):
        raise ReplyGenerationError(
            "The pickup line compared the profile owner to an animal or object."
        )
    return text


SMART_QUOTE_MAP = str.maketrans({
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
})


def prepare_reply_for_entry(text, max_length=MAX_REPLY_LENGTH):
    """Shape raw model text for Hinge entry without content validation."""
    text = str(text or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if lines:
        candidates = [
            line
            for line in lines
            if not re.match(
                r"^(?:reply|response|here(?:'s| is)|generated reply)\b",
                line,
                re.IGNORECASE,
            )
        ]
        text = max(candidates or lines, key=len)
    text = text.translate(SMART_QUOTE_MAP).strip().strip("\"'")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[:max_length].rstrip()
    return text


def extract_openai_response_text(response):
    text = getattr(response, "output_text", None)
    if text and str(text).strip():
        return str(text).strip()
    chunks = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            content_type = getattr(content, "type", None)
            if content_type in {"output_text", "text"}:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(str(value))
    return "\n".join(chunks).strip()


def validate_reply(reply, prompt_ids):
    if not isinstance(reply, GeneratedReply):
        raise ReplyGenerationError("The model returned an invalid reply object.")
    text = reply.reply.strip().translate(SMART_QUOTE_MAP)
    # iPhone Mirroring occasionally drops clipboard paste, while direct
    # keyboard entry is reliable. Remove decorative non-ASCII characters
    # (most commonly model-added emoji) so every accepted reply can use the
    # direct typing path.
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+([,.!?])", r"\1", text).strip()
    if reply.prompt_id not in set(prompt_ids):
        raise ReplyGenerationError("The model selected an unknown prompt.")
    if not text or "\n" in text or "\r" in text:
        raise ReplyGenerationError("The reply must be one non-empty line.")
    if len(text) > MAX_REPLY_LENGTH:
        raise ReplyGenerationError(f"The reply exceeds {MAX_REPLY_LENGTH} characters.")
    if re.search(r"\s['\"]$", text) or re.match(r"^['\"]\s", text):
        raise ReplyGenerationError("The reply contains an unmatched quote.")
    if re.search(r"[,;:\-–—]\s*$", text) or re.search(
        r"\b(?:and|but|for|or|to|with)\s*$",
        text,
        re.IGNORECASE,
    ):
        raise ReplyGenerationError("The reply appears truncated or unfinished.")
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    if normalized in GENERIC_REPLIES or len(normalized.split()) < 3:
        raise ReplyGenerationError("The reply is too generic.")
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in UNSAFE_PATTERNS):
        raise ReplyGenerationError("The reply failed local safety validation.")
    return GeneratedReply(reply.prompt_id, text)


def bind_single_prompt_id(reply, prompts):
    """Ignore a model-invented ID when exactly one known prompt is in scope."""
    if len(prompts) == 1 and isinstance(reply, GeneratedReply):
        return GeneratedReply(prompts[0].prompt_id, reply.reply)
    return reply


def grounding_words(prompt):
    return {
        word
        for word in re.sub(
            r"[^a-z0-9]+",
            " ",
            f"{prompt.prompt} {prompt.answer}".casefold(),
        ).split()
        if len(word) >= 4 and word not in GROUNDING_STOPWORDS
    }


def grounded_fallback(prompt, tone):
    """Produce a safe last-resort reply when the local model will not ground."""
    answer_words = [
        word
        for word in re.sub(r"[^a-z0-9]+", " ", prompt.answer.casefold()).split()
        if len(word) >= 4 and word not in GROUNDING_STOPWORDS
    ]
    if not answer_words:
        answer_words = sorted(grounding_words(prompt))
    if not answer_words:
        raise ReplyGenerationError(
            "The prompt has no concrete word available for a grounded reply."
        )
    word = max(answer_words, key=len)
    templates = {
        "Playful & clean": f"{word.capitalize()}? Bold choice—I respect the commitment.",
        "Flirty & bold": f"{word.capitalize()}? Bold choice. Convince me over a drink?",
        "Dry & clever": f"{word.capitalize()}. A surprisingly defensible position.",
    }
    reply = GeneratedReply(prompt.prompt_id, templates[tone])
    return validate_grounding(validate_reply(reply, [prompt.prompt_id]), [prompt])


def validate_grounding(reply, prompts):
    selected = next(
        (prompt for prompt in prompts if prompt.prompt_id == reply.prompt_id),
        None,
    )
    if selected is None:
        raise ReplyGenerationError("The model selected an unknown prompt.")
    source_words = grounding_words(selected)
    reply_words = set(
        re.sub(r"[^a-z0-9]+", " ", reply.reply.casefold()).split()
    )
    if source_words and not source_words.intersection(reply_words):
        raise ReplyGenerationError(
            "The reply did not reference a concrete word from the selected prompt answer."
        )
    return reply


def build_input(prompts, tone):
    if tone not in TONE_INSTRUCTIONS:
        raise ReplyGenerationError("Choose a supported humor tone.")
    prompt_rows = [
        {
            "prompt_id": item.prompt_id,
            "prompt": item.prompt,
            "answer": item.answer,
            "required_content_words": sorted(grounding_words(item)),
        }
        for item in prompts
    ]
    system = (
        "Write one funny Hinge comment grounded in exactly one supplied prompt/answer. "
        f"Tone: {TONE_INSTRUCTIONS[tone]} Keep it under {MAX_REPLY_LENGTH} characters and on one line. "
        "React to a concrete detail in that prompt's answer. The reply MUST reuse at least one "
        "distinctive content word exactly as written in the chosen prompt or answer. "
        "Do not write a generic reaction and do not respond to the prompt heading alone. "
        "Do not mention appearance, protected traits, sex, contact details, or the fact that AI wrote it. "
        "Use ASCII characters only: no emoji or decorative Unicode. "
        "Avoid generic greetings. Return only the requested structured result."
    )
    user = (
        "Choose the prompt whose answer has the best specific comedic hook, then reply directly to that answer:\n"
        + json.dumps(
        prompt_rows,
        ensure_ascii=False,
        )
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_paid_input(prompts, tone):
    """Build the compact, unguarded request used only by paid providers."""
    if not prompts:
        raise ReplyGenerationError("No profile prompts are available to generate from.")
    selected = prompts[0]
    return (
        f"Generate a {tone} reply for this dating profile prompt:\n"
        f"{selected.prompt + " " + selected.answer}\n\n"
        f"{PAID_HOUSE_RULES}"
    )


def build_paid_photo_input(tone):
    """Build the compact paid photo request with house rules."""
    if tone not in TONE_INSTRUCTIONS:
        raise ReplyGenerationError("Choose a supported humor tone.")
    return (
        f"Generate a {tone} reply to this dating profile photo.\n\n"
        f"{PAID_HOUSE_RULES}"
    )


def _png_image_b64(image_bytes):
    if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) < 64:
        raise ReplyGenerationError("The profile photo crop was empty or invalid.")
    return base64.b64encode(bytes(image_bytes)).decode("ascii")


def _paid_photo_reply_from_text(text):
    reply = prepare_reply_for_entry(text, max_length=PAID_MAX_REPLY_LENGTH)
    if not reply:
        raise ReplyGenerationError("The paid model returned no reply text.")
    return reply


REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt_id": {"type": "string"},
        "reply": {"type": "string", "maxLength": MAX_REPLY_LENGTH},
    },
    "required": ["prompt_id", "reply"],
    "additionalProperties": False,
}


def _generate_with_retries(
    request,
    prompts,
    tone,
    attempts,
    retry_exceptions,
    failure_prefix,
):
    if not prompts:
        raise ReplyGenerationError("No profile prompts are available to generate from.")
    prompt_ids = [prompt.prompt_id for prompt in prompts]
    last_error = None
    request_prompts = list(prompts)
    reply = None
    for _ in range(attempts):
        try:
            reply = bind_single_prompt_id(request(request_prompts, tone), prompts)
            return validate_grounding(validate_reply(reply, prompt_ids), prompts)
        except retry_exceptions as error:
            last_error = error
            if (
                isinstance(error, ReplyGenerationError)
                and "concrete word" in str(error)
                and reply is not None
            ):
                selected = next(
                    (prompt for prompt in prompts if prompt.prompt_id == reply.prompt_id),
                    None,
                )
                if selected is not None:
                    request_prompts = [selected]
    if isinstance(last_error, ReplyGenerationError):
        raise last_error
    raise ReplyGenerationError(f"{failure_prefix}: {last_error}") from last_error


def paid_model_from_selection(value):
    if not value:
        return None
    for model in PAID_MODELS:
        if value in {model.label, model.model_id}:
            return model
    return None


def parse_generated_reply(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ReplyGenerationError("The paid model returned malformed structured output.")
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as error:
            raise ReplyGenerationError(
                "The paid model returned malformed structured output."
            ) from error
    try:
        return GeneratedReply(str(payload["prompt_id"]), str(payload["reply"]))
    except (KeyError, TypeError) as error:
        raise ReplyGenerationError("The paid model returned malformed structured output.") from error


def _http_json(url, payload, headers, opener, timeout=20):
    data = json.dumps(payload).encode("utf-8")
    request = url_request.Request(url, data=data, headers=headers, method="POST")
    try:
        with opener(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except url_error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:240]
        raise ReplyGenerationError(
            f"{error.code} from the paid model API: {detail or error.reason}"
        ) from error
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ReplyGenerationError(f"The paid model request failed: {error}") from error


class ReplyGenerator:
    def __init__(self, client=None, model=PAY_MODEL, api_key=None):
        if client is None:
            key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
            if not key:
                raise ReplyGenerationError(
                    "Import or paste an OpenAI API key before using Prompt Reply."
                )
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ReplyGenerationError(
                    "Install the OpenAI SDK with: python -m pip install -r requirements.txt"
                ) from error
            client = OpenAI(api_key=key, timeout=60.0, max_retries=0)
        self.client = client
        self.model = model

    def model_input(self, prompts, tone):
        return build_paid_input(prompts, tone)

    def _request(self, prompts, tone):
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "low"},
            input=self.model_input(prompts, tone),
        )
        try:
            reply = prepare_reply_for_entry(
                extract_openai_response_text(response),
                max_length=PAID_MAX_REPLY_LENGTH,
            )
        except (AttributeError, IndexError) as error:
            raise ReplyGenerationError("OpenAI returned no reply text.") from error
        if not reply:
            raise ReplyGenerationError("OpenAI returned no reply text.")
        return GeneratedReply(prompt_id=prompts[0].prompt_id, reply=reply)

    def generate(self, prompts, tone):
        return self._request(prompts, tone)

    def generate_photo_pickup_line(self, image_bytes, tone):
        b64 = _png_image_b64(image_bytes)
        try:
            response = self.client.responses.create(
                model=self.model,
                reasoning={"effort": "low"},
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_paid_photo_input(tone)},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}",
                        },
                    ],
                }],
            )
        except Exception as error:
            raise ReplyGenerationError(
                f"OpenAI photo fallback failed: {error}"
            ) from error
        return _paid_photo_reply_from_text(extract_openai_response_text(response))


class CloudReplyGenerator:
    """Generate replies with Anthropic, Google, xAI, or DeepSeek."""

    def __init__(self, choice, api_key=None, opener=None):
        key = (api_key or os.environ.get(choice.env_key) or "").strip()
        if choice.provider == "google" and not key:
            key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
        if not key:
            raise ReplyGenerationError(
                f"Import or paste a {choice.provider_title} API key before using Prompt Reply."
            )
        self.choice = choice
        self.model = choice.model_id
        self.api_key = key
        self.opener = opener or url_request.urlopen

    def model_input(self, prompts, tone):
        return build_paid_input(prompts, tone)

    @staticmethod
    def _generated_reply(prompts, text):
        try:
            prompt_id = prompts[0].prompt_id
        except IndexError as error:
            raise ReplyGenerationError(
                "No profile prompts are available to generate from."
            ) from error
        reply = prepare_reply_for_entry(text, max_length=PAID_MAX_REPLY_LENGTH)
        if not reply:
            raise ReplyGenerationError("The paid model returned no reply text.")
        return GeneratedReply(prompt_id=prompt_id, reply=reply)

    def _openai_compatible_chat(self, url, prompts, tone):
        payload = _http_json(
            url,
            {
                "model": self.model,
                "temperature": 0.4,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": self.model_input(prompts, tone)}],
            },
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            self.opener,
        )
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ReplyGenerationError("The paid model returned malformed structured output.") from error

    def _request(self, prompts, tone):
        provider = self.choice.provider
        if provider == "anthropic":
            payload = _http_json(
                "https://api.anthropic.com/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": 256,
                    "messages": [
                        {"role": "user", "content": self.model_input(prompts, tone)}
                    ],
                },
                {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                self.opener,
            )
            try:
                text = payload["content"][0]["text"]
            except (KeyError, IndexError, TypeError) as error:
                raise ReplyGenerationError(
                    "The paid model returned no reply text."
                ) from error
            return self._generated_reply(prompts, text)
        if provider == "google":
            payload = _http_json(
                (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model}:generateContent?key={self.api_key}"
                ),
                {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": self.model_input(prompts, tone)}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 256,
                    },
                },
                {"Content-Type": "application/json"},
                self.opener,
            )
            try:
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as error:
                raise ReplyGenerationError(
                    "The paid model returned no reply text."
                ) from error
            return self._generated_reply(prompts, text)
        if provider == "xai":
            return self._generated_reply(
                prompts,
                self._openai_compatible_chat(
                    "https://api.x.ai/v1/chat/completions", prompts, tone
                ),
            )
        if provider == "deepseek":
            return self._generated_reply(
                prompts,
                self._openai_compatible_chat(
                    "https://api.deepseek.com/chat/completions",
                    prompts,
                    tone,
                )
            )
        raise ReplyGenerationError(f"Unsupported paid provider: {provider}")

    def generate(self, prompts, tone):
        return self._request(prompts, tone)

    def _photo_openai_compatible_chat(self, url, image_bytes, tone):
        b64 = _png_image_b64(image_bytes)
        payload = _http_json(
            url,
            {
                "model": self.model,
                "temperature": 0.4,
                "max_tokens": 256,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_paid_photo_input(tone)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }],
            },
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            self.opener,
        )
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ReplyGenerationError("The paid model returned no reply text.") from error

    def generate_photo_pickup_line(self, image_bytes, tone):
        provider = self.choice.provider
        b64 = _png_image_b64(image_bytes)
        prompt = build_paid_photo_input(tone)
        if provider == "deepseek":
            raise ReplyGenerationError("DeepSeek does not support photo fallback.")
        if provider == "anthropic":
            payload = _http_json(
                "https://api.anthropic.com/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": 256,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
                {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                self.opener,
            )
            try:
                text = payload["content"][0]["text"]
            except (KeyError, IndexError, TypeError) as error:
                raise ReplyGenerationError(
                    "The paid model returned no reply text."
                ) from error
            return _paid_photo_reply_from_text(text)
        if provider == "google":
            payload = _http_json(
                (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.model}:generateContent?key={self.api_key}"
                ),
                {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {"inlineData": {"mimeType": "image/png", "data": b64}},
                            {"text": prompt},
                        ],
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 256,
                    },
                },
                {"Content-Type": "application/json"},
                self.opener,
            )
            try:
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as error:
                raise ReplyGenerationError(
                    "The paid model returned no reply text."
                ) from error
            return _paid_photo_reply_from_text(text)
        if provider == "xai":
            return _paid_photo_reply_from_text(
                self._photo_openai_compatible_chat(
                    "https://api.x.ai/v1/chat/completions",
                    image_bytes,
                    tone,
                )
            )
        raise ReplyGenerationError(f"Unsupported paid provider: {provider}")


class OllamaReplyGenerator:
    """Generate replies with a local Ollama model and no API key."""

    def __init__(self, model=LOCAL_FREE_MODEL, base_url=OLLAMA_BASE_URL, opener=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.opener = opener or url_request.urlopen
        self._ready = False

    def _json_request(self, path, payload=None, timeout=5):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = url_request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with self.opener(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _find_executable():
        candidates = (
            shutil.which("ollama"),
            "/opt/homebrew/bin/ollama",
            "/usr/local/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        )
        return next((path for path in candidates if path and os.path.isfile(path)), None)

    def ensure_ready(self):
        if self._ready:
            return
        try:
            tags = self._json_request("/api/tags", timeout=1.5)
        except Exception:
            executable = self._find_executable()
            if not executable:
                raise ReplyGenerationError(
                    "Free Local mode requires Ollama. Install it from https://ollama.com/download."
                )
            subprocess.Popen(
                [executable, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # First launch after installation can take several seconds before
            # the local HTTP listener becomes ready.
            deadline = time.monotonic() + 30
            tags = None
            while time.monotonic() < deadline:
                try:
                    tags = self._json_request("/api/tags", timeout=1)
                    break
                except Exception:
                    time.sleep(0.25)
            if tags is None:
                raise ReplyGenerationError("Ollama could not be started on this Mac.")

        installed = {
            model.get("name") or model.get("model")
            for model in tags.get("models", [])
        }
        if self.model not in installed:
            raise ReplyGenerationError(
                f"Local model {self.model} is not installed. Run: ollama pull {self.model}"
            )
        self._ready = True

    def _request(self, prompts, tone):
        messages = build_input(prompts, tone)
        result = self._json_request(
            "/api/generate",
            {
                "model": self.model,
                "system": messages[0]["content"],
                "prompt": messages[1]["content"],
                "format": REPLY_SCHEMA,
                "think": False,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.65,
                    "num_ctx": 4096,
                    "num_predict": 70,
                },
            },
            timeout=90,
        )
        try:
            payload = json.loads(result["response"])
            return GeneratedReply(payload["prompt_id"], payload["reply"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ReplyGenerationError("The local model returned malformed output.") from error

    def _vision_chat(
        self,
        image_bytes,
        prompt,
        schema,
        timeout,
        temperature,
        num_predict,
        error_message="The local vision model returned malformed output.",
    ):
        result = self._json_request(
            "/api/chat",
            {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }],
                "format": schema,
                "think": False,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": temperature,
                    "num_ctx": 4096,
                    "num_predict": num_predict,
                },
            },
            timeout=timeout,
        )
        try:
            return json.loads(result["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ReplyGenerationError(error_message) from error

    def _photo_request(self, image_bytes, tone):
        if tone not in TONE_INSTRUCTIONS:
            raise ReplyGenerationError("Choose a supported humor tone.")
        prompt = (
            "Look only at the supplied dating-profile photo. First choose the main focal "
            "non-person detail. Prefer, in order: a clearly visible pet, primary activity, "
            "food, landmark, prominent object, then setting. Ignore incidental background "
            "clutter such as pipes, signs, furniture, and tiny objects. If no detail is clear, "
            "set usable to false and do not guess. Otherwise set usable to true and write one "
            "short pickup line about that detail. "
            f"Tone: {TONE_INSTRUCTIONS[tone]} "
            "Do not describe appearance, body, age, ethnicity, religion, disability, wealth, "
            "or attractiveness. Do not identify a person or infer relationships or location. "
            "Address the profile owner; never call or compare them to an animal or object. For "
            "a pet, say 'your cat' or 'your dog'. visual_detail should be a specific 2-6 word "
            "noun phrase, such as 'orange cat' rather than just 'cat'. "
            "The line must repeat at least one concrete word exactly from visual_detail, be "
            f"ASCII, one line, and at most {MAX_REPLY_LENGTH} characters."
        )
        return self._vision_chat(
            image_bytes,
            prompt,
            PHOTO_PICKUP_SCHEMA,
            timeout=90,
            temperature=0.30,
            num_predict=80,
        )

    def detect_written_prompt(self, image_bytes):
        """Extract the first visible written Hinge prompt for deterministic anchoring."""
        if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) < 64:
            raise ReplyGenerationError("The prompt viewport image was empty or invalid.")
        self.ensure_ready()
        prompt = (
            "Read this Hinge profile viewport. Find the first visible written prompt card "
            "from top to bottom. Return its prompt heading and the user's answer exactly as "
            "shown. Ignore profile metadata, buttons, captions, 'Let's get together', and "
            "other non-written cards. If a complete heading and answer are not both visible, "
            "set has_written_prompt to false and return empty strings. Do not infer missing text."
        )
        payload = self._vision_chat(
            bytes(image_bytes),
            prompt,
            PROMPT_VISION_SCHEMA,
            timeout=60,
            temperature=0.10,
            num_predict=120,
            error_message="The local vision model returned malformed prompt detection.",
        )
        try:
            if payload.get("has_written_prompt") is not True:
                return None
            prompt_text = str(payload["prompt"]).strip()
            answer_text = str(payload["answer"]).strip()
        except (KeyError, TypeError) as error:
            raise ReplyGenerationError(
                "The local vision model returned malformed prompt detection."
            ) from error
        if not prompt_text or not answer_text:
            return None
        return prompt_text, answer_text

    def generate_photo_pickup_line(self, image_bytes, tone):
        """Generate a locally grounded line from a cropped first-profile photo."""
        if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) < 64:
            raise ReplyGenerationError("The profile photo crop was empty or invalid.")
        self.ensure_ready()
        last_error = None
        for _ in range(2):
            try:
                payload = self._photo_request(bytes(image_bytes), tone)
                if payload.get("usable") is not True:
                    raise ReplyGenerationError(
                        "The vision model found no reliable non-sensitive photo detail."
                    )
                return validate_photo_pickup_line(
                    payload["line"],
                    payload["visual_detail"],
                )
            except (KeyError, ReplyGenerationError, OSError, url_error.URLError) as error:
                last_error = error
        raise ReplyGenerationError(
            f"Local photo pickup generation failed: {last_error}"
        ) from last_error

    def generate(self, prompts, tone):
        if not prompts:
            raise ReplyGenerationError("No profile prompts are available to generate from.")
        self.ensure_ready()
        try:
            return _generate_with_retries(
                self._request,
                prompts,
                tone,
                attempts=4,
                retry_exceptions=(ReplyGenerationError, OSError, url_error.URLError),
                failure_prefix="Local reply generation failed",
            )
        except ReplyGenerationError as error:
            if "concrete word" in str(error) and len(prompts) == 1:
                return grounded_fallback(prompts[0], tone)
            raise
