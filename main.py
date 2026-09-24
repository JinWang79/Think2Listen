#!/usr/bin/env python3
"""Run one Think2Listen interaction without starting a training job.

This entry point is a compact, readable demonstration of the paper pipeline:
an actor either asks for targeted audio evidence or returns a final answer.
It is not a benchmark or metric-computation script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI

from think2listen.prompts import ACTOR_SYSTEM_PROMPT, FINAL_DECISION_PROMPT, build_actor_user_prompt
from think2listen.rollout.environment import get_observation
from think2listen.rollout.protocol import parse_answer, parse_ask, parse_inline_action


MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024 * 1024


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


def validate_service_url(name: str, value: str) -> str:
    """Accept only explicitly allowed HTTP(S) model-service endpoints."""

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"{name} must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise SystemExit(f"{name} must not contain URL credentials")
    allowed = {
        host.strip().lower()
        for host in os.environ.get(
            "MODEL_SERVICE_ALLOWED_HOSTS", "127.0.0.1,localhost,::1"
        ).split(",")
        if host.strip()
    }
    if parsed.hostname.lower() not in allowed:
        raise SystemExit(
            f"{name} host is not in MODEL_SERVICE_ALLOWED_HOSTS: {parsed.hostname}"
        )
    return value.rstrip("/")


def load_record(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise SystemExit("--index must be non-negative")
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise SystemExit(f"Input exceeds {MAX_INPUT_BYTES} bytes")
    selected = None
    seen = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            if len(raw_line) > MAX_LINE_BYTES:
                raise SystemExit(f"JSONL line exceeds {MAX_LINE_BYTES} bytes")
            if not raw_line.strip():
                continue
            if seen == index:
                selected = json.loads(raw_line.decode("utf-8"))
                break
            seen += 1
    if selected is None:
        raise SystemExit(f"No non-empty JSONL row at index {index}")
    if not isinstance(selected, dict):
        raise SystemExit("Selected JSONL row must be an object")
    return selected


def normalize_sample(record: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}

    def field(name: str) -> Any:
        return metadata.get(name, record.get(name))

    choices = field("choices")
    if isinstance(choices, list):
        choices = {chr(ord("A") + offset): value for offset, value in enumerate(choices)}
    if not isinstance(choices, dict):
        raise SystemExit("Sample choices must be a mapping or a list")
    sample = {
        "sample_id": str(field("sample_id") or f"sample-{index}"),
        "question": str(field("question") or "").strip(),
        "choices": {str(key).upper(): str(value).strip() for key, value in choices.items()},
        "initial_description": str(field("initial_description") or "").strip(),
        "audio_path": str(field("audio_path") or "").strip(),
    }
    missing = [name for name in ("question", "initial_description", "audio_path") if not sample[name]]
    if missing:
        raise SystemExit(f"Sample is missing required fields: {', '.join(missing)}")
    return sample


def parse_action(text: str, *, thinking_enabled: bool) -> tuple[str, str] | None:
    ask = parse_ask(text)
    if ask is not None:
        return "ask", ask
    answer = parse_answer(text)
    if answer is not None:
        return "answer", answer
    return parse_inline_action(text) if thinking_enabled else None


async def actor_turn(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    thinking_enabled: bool,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking_enabled}},
    )
    return (response.choices[0].message.content or "").strip()


async def run_interaction(args: argparse.Namespace) -> dict[str, Any]:
    record = load_record(args.input, args.index)
    sample = normalize_sample(record, args.index)
    thinking_enabled = os.environ.get("ACTOR_ENABLE_THINKING", "1") == "1"
    actor_base_url = validate_service_url(
        "ACTOR_MODEL_BASE_URL", required_env("ACTOR_MODEL_BASE_URL")
    )
    audio_base_url = os.environ.get("AUDIO_MODEL_BASE_URL", "http://127.0.0.1:8901/v1")
    os.environ["AUDIO_MODEL_BASE_URL"] = validate_service_url(
        "AUDIO_MODEL_BASE_URL", audio_base_url
    )
    actor = AsyncOpenAI(
        base_url=actor_base_url,
        api_key=os.environ.get("ACTOR_MODEL_API_KEY", "EMPTY"),
        timeout=float(os.environ.get("ACTOR_MODEL_TIMEOUT_SECONDS", "120")),
        max_retries=0,
    )
    model = required_env("ACTOR_MODEL_NAME_OR_PATH")
    if not args.mock_audio:
        required_env("AUDIO_MODEL_NAME_OR_PATH")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_actor_user_prompt(
                question=sample["question"],
                choices=sample["choices"],
                initial_description=sample["initial_description"],
                interactions=None,
            ),
        },
    ]
    transcript: list[dict[str, Any]] = []
    final_answer = None
    status = "missing_answer"

    for decision in range(args.max_ask_steps + 1):
        forced_answer = len(transcript) >= args.max_ask_steps
        if forced_answer:
            messages.append({"role": "user", "content": FINAL_DECISION_PROMPT})
        text = await actor_turn(
            actor,
            model=model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            thinking_enabled=thinking_enabled,
        )
        action = parse_action(text, thinking_enabled=thinking_enabled)
        if action is None:
            status = "invalid_action"
            transcript.append({"turn": decision + 1, "kind": "invalid", "actor": text})
            break

        kind, value = action
        if kind == "answer":
            if value not in sample["choices"]:
                status = "invalid_answer_label"
                transcript.append({
                    "turn": decision + 1,
                    "kind": "invalid",
                    "actor": text,
                    "answer": value,
                })
                break
            final_answer = value
            status = "answered" if len(transcript) >= args.min_ask_steps else "answer_before_min_asks"
            transcript.append({"turn": decision + 1, "kind": "answer", "actor": text, "answer": value})
            break
        if forced_answer:
            status = "ask_after_max"
            transcript.append({"turn": decision + 1, "kind": "ask_after_max", "actor": text, "query": value})
            break

        if args.mock_audio:
            observation = f"MOCK_AUDIO_TURN_{len(transcript) + 1}: received query: {value}"
        else:
            observation = await get_observation(
                sample_id=sample["sample_id"],
                audio_path=sample["audio_path"],
                query=value,
            )
        transcript.append({
            "turn": decision + 1,
            "kind": "ask",
            "actor": text,
            "query": value,
            "observation": observation,
        })
        messages.extend([
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"[Audio Observation]\n{observation}"},
        ])

    return {
        "sample_id": sample["sample_id"],
        "status": status,
        "final_answer": final_answer,
        "ask_count": sum(item["kind"] == "ask" for item in transcript),
        "trajectory": transcript,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prompt JSONL file")
    parser.add_argument("--index", type=int, default=0, help="Zero-based non-empty row index")
    parser.add_argument("--output", type=Path, help="Optional JSON result path; defaults to stdout")
    parser.add_argument("--min-ask-steps", type=int, default=0)
    parser.add_argument("--max-ask-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--mock-audio", action="store_true", help="Show the control flow without calling the audio model")
    args = parser.parse_args()
    if args.min_ask_steps < 0 or args.max_ask_steps < max(1, args.min_ask_steps):
        parser.error("require 0 <= min-ask-steps <= max-ask-steps and max-ask-steps >= 1")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_interaction(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
