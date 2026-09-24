"""Action parsing for the end-to-end active-audio policy."""

from __future__ import annotations

import re

from think2listen.rollout.cache import normalize_query


ASK_PATTERN = re.compile(r"^\s*<ask>\s*(.*?)\s*</ask>\s*$", re.DOTALL)
VALID_ANSWER_LABELS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ANSWER_PATTERN = re.compile(r"^\s*<answer>\s*([A-Z])\s*</answer>\s*$", re.DOTALL)
TRAILING_CHAT_TOKENS = re.compile(r"(?:\s*<\|(?:im_end|end_of_text)\|>\s*)+$")
# The action body must not cross another tag opener. This skips an unclosed
# action-like string in a thought block and finds the later complete action.
INLINE_ACTION_PATTERN = re.compile(r"<(ask|answer)>\s*([^<]*?)\s*</\1>", re.DOTALL | re.IGNORECASE)


def strip_trailing_chat_tokens(text: str) -> str:
    return TRAILING_CHAT_TOKENS.sub("", text or "")


def parse_ask(text: str) -> str | None:
    match = ASK_PATTERN.match(strip_trailing_chat_tokens(text))
    return normalize_query(match.group(1)) if match else None


def parse_answer(text: str) -> str | None:
    match = ANSWER_PATTERN.match(strip_trailing_chat_tokens(text))
    return match.group(1) if match else None


def parse_inline_action(text: str) -> tuple[str, str] | None:
    """Extract the final well-formed action from a thinking-mode continuation."""
    matches = list(INLINE_ACTION_PATTERN.finditer(strip_trailing_chat_tokens(text)))
    if not matches:
        return None
    match = matches[-1]
    kind = match.group(1).lower()
    value = match.group(2).strip()
    if kind == "answer":
        return (kind, value.upper()) if value.upper() in VALID_ANSWER_LABELS else None
    query = normalize_query(value)
    return (kind, query) if query else None
