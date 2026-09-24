"""Gold-blind candidate-state extraction for Transition-as-Credit."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any

from openai import OpenAI

from think2listen.prompts import (
    CANDIDATE_STATE_EXTRACTOR_SYSTEM_PROMPT,
    CANDIDATE_STATE_EXTRACTOR_TEMPLATE,
    format_choices,
)


EXTRACTOR_BASE_URL = os.environ.get(
    "TRANSITION_CREDIT_EXTRACTOR_BASE_URL", "http://127.0.0.1:8006/v1"
)
EXTRACTOR_MODEL = os.environ.get("TRANSITION_CREDIT_EXTRACTOR_MODEL", "").strip()
EXTRACTOR_MAX_TOKENS = int(os.environ.get("TRANSITION_CREDIT_EXTRACTOR_MAX_TOKENS", "8192"))
EXTRACTOR_TIMEOUT_SECONDS = float(os.environ.get("TRANSITION_CREDIT_EXTRACTOR_TIMEOUT_SECONDS", "90"))
EXTRACTOR_ATTEMPTS = int(os.environ.get("TRANSITION_CREDIT_EXTRACTOR_ATTEMPTS", "2"))
EXTRACTOR_ENABLE_THINKING = os.environ.get(
    "TRANSITION_CREDIT_EXTRACTOR_ENABLE_THINKING", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EXTRACTOR_MAX_CONCURRENCY = int(os.environ.get("TRANSITION_CREDIT_EXTRACTOR_MAX_CONCURRENCY", "64"))

ASK_BLOCK_PATTERN = re.compile(
    r"\[ASK\s+(\d+)\]\s*(.*?)(?=\n\[AUDIO OBSERVATION\])",
    re.IGNORECASE | re.DOTALL,
)
_AUDIT_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "transition_credit_audit_context", default=None
)
_AUDIT_LOCK = Lock()
_EXECUTOR = ThreadPoolExecutor(
    max_workers=EXTRACTOR_MAX_CONCURRENCY,
    thread_name_prefix="transition-credit-extractor",
)
_SEMAPHORE: asyncio.Semaphore | None = None


def _metadata(sample: Any) -> dict[str, Any]:
    return dict((sample.get("metadata") if isinstance(sample, dict) else sample.metadata) or {})


def _response(sample: Any) -> str:
    return str(sample.get("response") if isinstance(sample, dict) else getattr(sample, "response", "") or "")


def _extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _message_text(message: Any) -> str:
    return str(
        getattr(message, "content", None)
        or getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    ).strip()


def _prior_history(trajectory: list[dict[str, Any]], turn: int) -> str:
    prior = trajectory[: turn - 1]
    if not prior:
        return "None."
    return "\n\n".join(
        f"Turn {index} ASK: {str(item.get('query') or '').strip()}\n"
        f"Turn {index} audio observation: {str(item.get('observation') or '').strip()}"
        for index, item in enumerate(prior, 1)
    )


def _actor_think_for_turn(response: str, turn: int) -> str:
    for match in ASK_BLOCK_PATTERN.finditer(response or ""):
        if int(match.group(1)) != turn:
            continue
        action_text = match.group(2).strip()
        ask_start = action_text.lower().rfind("<ask>")
        reasoning = action_text[:ask_start].strip() if ask_start >= 0 else action_text
        return reasoning or "None provided."
    return "None provided."


def _validate(payload: Any, labels: set[str]) -> list[str]:
    if not isinstance(payload, dict):
        return ["top-level JSON value is not an object"]
    errors: list[str] = []
    if str(payload.get("ask_is_specific") or "") not in {"yes", "no"}:
        errors.append("ask_is_specific must be yes or no")
    values: dict[str, set[str]] = {}
    for key in ("plausible_before", "plausible_after", "eliminated_distractors"):
        raw = payload.get(key)
        if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
            errors.append(f"{key} must be a list of labels")
            continue
        values[key] = set(raw)
        unknown = values[key] - labels
        if unknown:
            errors.append(f"{key} contains unknown labels: {sorted(unknown)}")
    before = values.get("plausible_before", set())
    after = values.get("plausible_after", set())
    eliminated = values.get("eliminated_distractors", set())
    if after - before:
        errors.append("plausible_after must be a subset of plausible_before")
    if eliminated - before:
        errors.append("eliminated_distractors must be plausible_before")
    if eliminated & after:
        errors.append("eliminated_distractors cannot remain plausible_after")
    if not str(payload.get("reason") or "").strip():
        errors.append("reason is empty")
    return errors


def _write_audit(record: dict[str, Any]) -> None:
    path = os.environ.get("TRANSITION_CREDIT_EXTRACTOR_AUDIT_PATH", "").strip()
    if not path:
        return
    try:
        audit_path = Path(path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        pass


def _judge_one(prompt: str, labels: set[str]) -> tuple[dict[str, Any], int]:
    if not EXTRACTOR_MODEL:
        raise RuntimeError("TRANSITION_CREDIT_EXTRACTOR_MODEL must be configured")
    correction = ""
    last_errors = ["unknown error"]
    audit = {
        **(_AUDIT_CONTEXT.get() or {}),
        "protocol": "gold_blind_candidate_extractor_v1",
        "model": EXTRACTOR_MODEL,
        "timeout_seconds": EXTRACTOR_TIMEOUT_SECONDS,
        "max_tokens": EXTRACTOR_MAX_TOKENS,
        "prompt": prompt,
        "attempts": [],
    }
    try:
        with OpenAI(
            base_url=EXTRACTOR_BASE_URL,
            api_key=os.environ.get("TRANSITION_CREDIT_EXTRACTOR_API_KEY", "EMPTY"),
            timeout=EXTRACTOR_TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            for attempt in range(1, EXTRACTOR_ATTEMPTS + 1):
                try:
                    response = client.chat.completions.create(
                        model=EXTRACTOR_MODEL,
                        messages=[
                            {"role": "system", "content": CANDIDATE_STATE_EXTRACTOR_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt + correction},
                        ],
                        temperature=0.0,
                        top_p=0.95,
                        max_tokens=EXTRACTOR_MAX_TOKENS,
                        extra_body={"chat_template_kwargs": {"enable_thinking": EXTRACTOR_ENABLE_THINKING}},
                    )
                    message = response.choices[0].message
                    payload = _extract_json(_message_text(message))
                    last_errors = _validate(payload, labels)
                    audit["attempts"].append(
                        {"attempt": attempt, "parsed_payload": payload, "validation_errors": last_errors}
                    )
                    if not last_errors:
                        audit["status"] = "ok"
                        audit["final_extraction"] = payload
                        return payload, attempt
                    correction = (
                        "\n\nYour previous JSON failed validation: "
                        + "; ".join(last_errors)
                        + ". Return corrected JSON only."
                    )
                except Exception as exc:
                    last_errors = [repr(exc)]
                    audit["attempts"].append({"attempt": attempt, "error": repr(exc)})
                if attempt < EXTRACTOR_ATTEMPTS:
                    time.sleep(0.5)
        raise RuntimeError("; ".join(last_errors))
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = repr(exc)
        raise
    finally:
        _write_audit(audit)


def _prompt_from_parts(
    metadata: dict[str, Any],
    trajectory: list[dict[str, Any]],
    turn: int,
    actor_reasoning: str,
) -> str:
    item = trajectory[turn - 1]
    return CANDIDATE_STATE_EXTRACTOR_TEMPLATE.format(
        question=str(metadata.get("question") or ""),
        choices=format_choices(dict(metadata.get("choices") or {})),
        initial_description=str(metadata.get("initial_description") or "None provided."),
        prior_history=_prior_history(trajectory, turn),
        actor_reasoning=actor_reasoning,
        current_ask=str(item.get("query") or ""),
        current_observation=str(item.get("observation") or "None returned."),
    )


async def _run_extractor_turn(
    prompt: str, labels: set[str], context: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(EXTRACTOR_MAX_CONCURRENCY)
    async with _SEMAPHORE:
        def run() -> tuple[dict[str, Any], int]:
            token = _AUDIT_CONTEXT.set(context)
            try:
                return _judge_one(prompt, labels)
            finally:
                _AUDIT_CONTEXT.reset(token)

        return await asyncio.get_running_loop().run_in_executor(_EXECUTOR, run)


def _is_misleading_transition(extraction: dict[str, Any], gold: str) -> bool:
    before = set(extraction["plausible_before"])
    after = set(extraction["plausible_after"])
    return (
        extraction["ask_is_specific"] == "yes"
        and len(before) == 2
        and gold in before
        and len(after) == 1
        and gold not in after
    )


async def score_runtime_sample(args: Any, sample: Any) -> dict[str, Any]:
    """Reconstruct all ASK transitions while keeping gold hidden from the model."""

    del args
    metadata = _metadata(sample)
    trajectory = list(metadata.get("e2e_trajectory") or [])
    if not trajectory:
        return {
            "status": "no_ask",
            "protocol": "transition_as_credit_v1",
            "ask_count": 0,
            "misleading": False,
            "steps": [],
        }
    labels = set(str(key) for key in dict(metadata.get("choices") or {}))
    gold = str(metadata.get("correct_letter") or "")
    prompts = [
        _prompt_from_parts(metadata, trajectory, turn, _actor_think_for_turn(_response(sample), turn))
        for turn in range(1, len(trajectory) + 1)
    ]
    context = {
        "sample_id": metadata.get("sample_id"),
        "group_index": getattr(sample, "group_index", None),
        "sample_index": getattr(sample, "index", None),
    }
    results = await asyncio.gather(
        *(
            _run_extractor_turn(
                prompt,
                labels,
                {**context, "turn": turn, "query": trajectory[turn - 1].get("query")},
            )
            for turn, prompt in enumerate(prompts, 1)
        ),
        return_exceptions=True,
    )
    misleading = False
    steps: list[dict[str, Any]] = []
    for turn, result in enumerate(results, 1):
        if isinstance(result, Exception):
            steps.append(
                {
                    "turn": turn,
                    "query": trajectory[turn - 1].get("query"),
                    "extractor_error": repr(result),
                    "misleading": False,
                }
            )
            continue
        extraction, attempts = result
        step_misleading = _is_misleading_transition(extraction, gold)
        misleading = misleading or step_misleading
        steps.append(
            {
                "turn": turn,
                "query": trajectory[turn - 1].get("query"),
                "attempts": attempts,
                "extraction": extraction,
                "misleading": step_misleading,
            }
        )
    return {
        "status": "ok",
        "protocol": "transition_as_credit_v1",
        "ask_count": len(trajectory),
        "misleading": misleading,
        "steps": steps,
    }
