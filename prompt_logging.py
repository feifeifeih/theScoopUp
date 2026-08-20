"""Formatting and file naming helpers for prompt-reply logs."""

import re
from datetime import datetime, timezone
from pathlib import Path

from reply_generation import API_KEY_ENV_NAMES


PROMPT_TRANSCRIPT_DIR = Path.home() / "Desktop"


def parse_openai_api_key(text, env_names=API_KEY_ENV_NAMES):
    """Read an API key from pasted text or a small env/key file."""
    if not isinstance(text, str):
        return ""
    allowed = {name for name in env_names} | {name.lower() for name in env_names}
    for raw in text.splitlines():
        line = raw.strip().strip("'\"")
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" in line:
            name, value = line.split("=", 1)
            if name.strip() not in allowed:
                continue
            line = value.strip().strip("'\"")
        if line:
            return line
    return ""


def prompt_transcript_path(when=None, model="unknown-model"):
    """Desktop text log named Scoop plus the model and local start time."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H-%M-%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model)).strip("-")
    return PROMPT_TRANSCRIPT_DIR / f"Scoop {safe_model or 'unknown-model'} {stamp}.txt"


def format_elapsed_time(seconds):
    """Format a monotonic duration for compact status messages."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_prompt_transcript_header(*, batch_id, tone, engine, model, when=None):
    """Render batch metadata once at the top of a transcript log file."""
    stamp = (when or datetime.now(timezone.utc)).isoformat()
    return "\n".join([
        "======== The Scoop UP Prompt Reply Log ========",
        f"Batch: {batch_id}",
        f"Started: {stamp}",
        f"Engine: {engine or '(none)'}",
        f"Model: {model or '(none)'}",
        f"Tone: {tone or '(none)'}",
        "",
    ]) + "\n"


def format_prompt_transcript(record):
    """Render one profile transcript as plain text for the Desktop log."""
    def section(title, value):
        text = "" if value is None else str(value).strip()
        return f"{title}:\n{text or '(none)'}"

    def model_input_text(value):
        if not value:
            return "(none)"
        if isinstance(value, list):
            chunks = []
            for item in value:
                if isinstance(item, dict):
                    role = str(item.get("role") or "message").strip()
                    content = str(item.get("content") or "").strip()
                    chunks.append(f"[{role}]\n{content}")
                else:
                    chunks.append(str(item).strip())
            return "\n\n".join(chunk for chunk in chunks if chunk) or "(none)"
        return str(value).strip() or "(none)"

    lines = [
        (
            f"======== Profile {record.get('rotation')}  "
            f"{record.get('timestamp')}  {record.get('outcome')}  ========"
        ),
        section("Prompt", record.get("profile_prompt")),
        "",
        section("Answer", record.get("profile_answer")),
        "",
        section("Sent to model", model_input_text(record.get("model_input"))),
        "",
    ]
    if record.get("outcome") == "sent":
        lines.append(section("Reply sent to profile", record.get("sent_to_profile")))
    else:
        lines.append(f"Stage: {record.get('stage') or '(none)'}")
        lines.append(section("Error", record.get("error")))
        lines.append("")
        lines.append(section("Reply (not sent)", record.get("model_reply")))
    lines.append("")
    return "\n".join(lines) + "\n"
