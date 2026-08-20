"""Prompt parsing, matching, and reply visibility checks."""

from difflib import SequenceMatcher
import hashlib
import math
import re

from profile_vision import MIN_OCR_CONFIDENCE, CapturedPrompt, normalize_text


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
