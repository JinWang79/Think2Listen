"""Load and compact-tokenize verified teacher trajectories."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from think2listen.prompts import ACTOR_SYSTEM_PROMPT, build_actor_user_prompt


class TeacherTrajectoryTooLongError(RuntimeError):
    """Raised when a tokenized teacher row exceeds the configured hard limit."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_fraction(value: str) -> float:
    number = int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)
    return number / float(16**16)


def _bucket_weights() -> dict[str, float]:
    value = os.environ.get(
        "TEACHER_TRAJECTORY_BUCKET_WEIGHTS", "direct:0.10,one_ask:0.35,multi_ask:0.55"
    )
    result = {}
    for item in value.split(","):
        name, weight = item.split(":", 1)
        result[name.strip()] = float(weight)
    expected = {"direct", "one_ask", "multi_ask"}
    if set(result) != expected or any(weight < 0 for weight in result.values()) or sum(result.values()) <= 0:
        raise ValueError(f"Invalid TEACHER_TRAJECTORY_BUCKET_WEIGHTS: {value!r}")
    return result


@lru_cache(maxsize=2)
def get_tokenizer(path: str | None = None):
    resolved = path or os.environ.get("TEACHER_TRAJECTORY_TOKENIZER_PATH", "").strip()
    if not resolved:
        raise RuntimeError("TEACHER_TRAJECTORY_TOKENIZER_PATH must be configured")
    return AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)


def _apply_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )
    while True:
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
            continue
        if isinstance(token_ids, (list, tuple)) and len(token_ids) == 1 and isinstance(
            token_ids[0], (Mapping, list, tuple)
        ):
            token_ids = token_ids[0]
            continue
        break
    return [int(token_id) for token_id in token_ids]


def _encode_action(tokenizer, action: dict[str, Any], target_policy: str) -> tuple[list[int], list[int]]:
    canonical = str(action["canonical_action"])
    if target_policy == "raw_teacher_actions":
        text = str(action.get("raw_teacher_text") or "")
        if not text.endswith(canonical):
            raise RuntimeError(f"Raw teacher action does not end with {canonical!r}")
    elif target_policy == "canonical_actions_only":
        text = f"</think>\n\n{canonical}"
    else:
        raise RuntimeError(f"Unknown teacher target policy: {target_policy!r}")
    start = text.rfind(canonical)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    if target_policy == "raw_teacher_actions":
        mask = [1] * len(ids)
    else:
        mask = [int(end > start and begin < len(text)) for begin, end in encoded["offset_mapping"]]
    if not ids or not any(mask):
        raise RuntimeError(f"Failed to tokenize canonical teacher action: {canonical!r}")
    return ids, mask


def build_teacher_sample(row: dict[str, Any], tokenizer=None) -> dict[str, Any]:
    tokenizer = tokenizer or get_tokenizer()
    target_policy = str(row.get("training_target_policy") or "canonical_actions_only")
    if target_policy == "raw_teacher_actions" and (
        row.get("same_actor_prompt_as_rl") is not True
        or row.get("prompt_alignment") != "public_v1_exact"
    ):
        raise RuntimeError("Only prompt-aligned natural trajectories may supervise raw teacher reasoning")
    base_ids = _apply_template(
        tokenizer,
        [
            {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": build_actor_user_prompt(
                question=str(row["question"]),
                choices={str(key): str(value) for key, value in dict(row["choices"]).items()},
                initial_description=str(row["initial_description"]),
                interactions=None,
            )},
        ],
        add_generation_prompt=True,
    )
    tokens = list(base_ids)
    full_mask = [0] * len(base_ids)
    masked_hint_actions = 0
    masked_hint_tokens = 0
    for action_index, action in enumerate(row["actions"]):
        action_ids, action_mask = _encode_action(tokenizer, action, target_policy)
        # The privileged rescue hint is conditioning context, not a teacher CE
        # target. All subsequent natural actor thinking/actions stay scored.
        if action.get("teacher_ce_mask") is False:
            masked_hint_actions += 1
            masked_hint_tokens += sum(action_mask)
            action_mask = [0] * len(action_mask)
        tokens.extend(action_ids)
        full_mask.extend(action_mask)
        if action["kind"] == "ask":
            observation = str(action.get("observation") or "").strip()
            if not observation:
                raise RuntimeError(f"Teacher trajectory {row['trajectory_id']} has an empty observation")
            environment_ids = _apply_template(
                tokenizer,
                [{"role": "user", "content": f"[Audio Observation]\n{observation}"}],
                add_generation_prompt=True,
            )
            tokens.extend(environment_ids)
            full_mask.extend([0] * len(environment_ids))
    response_length = len(tokens) - len(base_ids)
    response_mask = full_mask[len(base_ids):]
    if len(response_mask) != response_length or not any(response_mask):
        raise RuntimeError(f"Invalid compact teacher mask for {row['trajectory_id']}")
    max_total = int(os.environ.get("TEACHER_TRAJECTORY_MAX_TOTAL_TOKENS", "8192"))
    if len(tokens) > max_total:
        raise TeacherTrajectoryTooLongError(
            f"Compact teacher trajectory {row['trajectory_id']} has {len(tokens)} > {max_total} tokens"
        )
    return {
        "tokens": tokens,
        "response_length": response_length,
        "loss_mask": response_mask,
        "teacher_marker": [1.0] * response_length,
        "trajectory_id": row["trajectory_id"],
        "sample_id": row["sample_id"],
        "ask_count": row["ask_count"],
        "ask_bucket": row["ask_bucket"],
        "training_target_policy": target_policy,
        "target_token_count": sum(response_mask),
        "total_token_count": len(tokens),
        "masked_hint_actions": masked_hint_actions,
        "masked_hint_token_count": masked_hint_tokens,
    }


# Kept for compatibility with the initial experiment's smoke tooling.
build_compact_teacher_sample = build_teacher_sample


class TeacherBank:
    def __init__(self, path: str | Path | None = None):
        resolved = path or os.environ.get("TEACHER_TRAJECTORY_BANK_PATH", "").strip()
        if not resolved:
            raise RuntimeError("TEACHER_TRAJECTORY_BANK_PATH must be configured")
        self.path = Path(resolved)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Teacher bank not found: {self.path}. Build or provide a verified trajectory bank first."
            )
        required_audio_version = os.environ.get(
            "TEACHER_TRAJECTORY_REQUIRED_AUDIO_PROMPT_VERSION", "targeted-audio-v1"
        )
        self.rows = _read_jsonl(self.path)
        self.by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            if row.get("data_split") not in (None, "train"):
                raise RuntimeError(f"Non-train teacher row found in training bank: {row['trajectory_id']}")
            if row.get("actions", [])[-1].get("answer") != row.get("correct_letter"):
                raise RuntimeError(f"Teacher trajectory {row['trajectory_id']} is not verified correct")
            if row.get("ask_count", 0) and required_audio_version:
                row_audio_version = row.get("audio_prompt_version", row.get("omni_prompt_version"))
                if row_audio_version != required_audio_version:
                    raise RuntimeError(
                        f"Teacher trajectory {row['trajectory_id']} uses audio prompt "
                        f"{row_audio_version!r}, expected {required_audio_version!r}"
                    )
            self.by_sample[str(row["sample_id"])].append(row)

    def choose(self, sample_id: str, group_index: int | str) -> dict[str, Any] | None:
        candidates = self.by_sample.get(str(sample_id), [])
        if not candidates:
            return None
        # A unique verified trajectory needs no bucket policy; this also keeps
        # legacy/new bucket labels compatible for single-candidate banks.
        if len(candidates) == 1:
            return candidates[0]
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_bucket[str(row["ask_bucket"])].append(row)
        weights = _bucket_weights()
        available = sorted(by_bucket)
        total = sum(weights[bucket] for bucket in available)
        point = _stable_fraction(f"bucket\0{sample_id}\0{group_index}") * total
        chosen_bucket = available[-1]
        cumulative = 0.0
        for bucket in available:
            cumulative += weights[bucket]
            if point <= cumulative:
                chosen_bucket = bucket
                break
        bucket_rows = sorted(by_bucket[chosen_bucket], key=lambda row: row["trajectory_id"])
        index = int(_stable_fraction(f"trajectory\0{sample_id}\0{group_index}") * len(bucket_rows))
        return bucket_rows[min(index, len(bucket_rows) - 1)]


@lru_cache(maxsize=1)
def get_teacher_bank() -> TeacherBank:
    return TeacherBank()

