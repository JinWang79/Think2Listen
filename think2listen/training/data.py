"""Unified training-data conversion with two independently controlled features."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from think2listen.rollout.protocol import parse_answer, parse_inline_action
from think2listen.transition_credit.reward_postprocess import post_process


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _flatten(samples: list[Any]) -> list[Any]:
    return [item for group in samples for item in group] if samples and isinstance(samples[0], list) else list(samples)


def _is_correct(sample: Any) -> bool:
    metadata = sample.metadata or {}
    final_action = str(metadata.get("e2e_final_action") or "")
    answer = parse_answer(final_action)
    if answer is None and metadata.get("e2e_actor_enable_thinking"):
        inline = parse_inline_action(final_action)
        if inline is not None and inline[0] == "answer":
            answer = inline[1]
    return (
        metadata.get("e2e_terminal_status") == "answered"
        and answer is not None
        and answer == str(metadata.get("correct_letter"))
    )


def _write_audit(record: dict[str, Any]) -> None:
    audit_path = os.environ.get("TEACHER_TRAJECTORY_AUDIT_PATH", "").strip()
    if not audit_path:
        return
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _assign_rollout_ids(flat: list[Any]) -> list[Any]:
    rollout_ids = [sample.rollout_id for sample in flat]
    used = {rollout_id for rollout_id in rollout_ids if rollout_id is not None}
    next_id = 0
    for index, rollout_id in enumerate(rollout_ids):
        if rollout_id is not None:
            continue
        while next_id in used:
            next_id += 1
        rollout_ids[index] = next_id
        used.add(next_id)
    return rollout_ids


def convert_samples_to_train_data(args: Any, samples: list[Any]) -> dict[str, Any]:
    """Calibrate rewards, then optionally replace one all-negative slot."""

    flat = _flatten(samples)
    group_size = int(args.n_samples_per_prompt)
    if not flat or len(flat) % group_size:
        raise RuntimeError("Training conversion requires complete rollout groups")
    if float(os.environ.get("ASK_COST", "0")) != 0 or float(os.environ.get("ASK_BONUS", "0")) != 0:
        raise RuntimeError("The paper configuration requires ASK_COST=0 and ASK_BONUS=0")

    raw_rewards, rewards = post_process(args, flat)
    if len(raw_rewards) != len(flat) or len(rewards) != len(flat):
        raise RuntimeError("reward conversion length mismatch")

    loss_masks: list[list[int]] = []
    for sample in flat:
        mask = list(sample.loss_mask) if sample.loss_mask is not None else [1] * int(sample.response_length)
        if len(mask) != int(sample.response_length):
            raise RuntimeError("rollout loss mask and response length differ")
        loss_masks.append([0] * len(mask) if sample.remove_sample else mask)

    rollout_ids = _assign_rollout_ids(flat)
    data: dict[str, Any] = {
        "rollout_ids": rollout_ids,
        "tokens": [list(sample.tokens) for sample in flat],
        "response_lengths": [int(sample.response_length) for sample in flat],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in flat],
        "sample_indices": [sample.index for sample in flat],
        "loss_masks": loss_masks,
        "teacher_log_probs": [[0.0] * int(sample.response_length) for sample in flat],
    }
    if flat[0].rollout_log_probs is not None:
        data["rollout_log_probs"] = [list(sample.rollout_log_probs) for sample in flat]

    replacements: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    all_negative_groups = 0
    bank_hits = 0
    if _enabled("ENABLE_TEACHER_TRAJECTORIES"):
        from think2listen.teacher_trajectories.bank import (
            TeacherTrajectoryTooLongError,
            build_teacher_sample,
            get_teacher_bank,
            get_tokenizer,
        )

        bank = get_teacher_bank()
        tokenizer = get_tokenizer()
        for start in range(0, len(flat), group_size):
            end = start + group_size
            group = flat[start:end]
            if any(sample.remove_sample for sample in group) or any(_is_correct(sample) for sample in group):
                continue
            all_negative_groups += 1
            sample_ids = {str((sample.metadata or {}).get("sample_id")) for sample in group}
            group_indices = {sample.group_index for sample in group}
            if len(sample_ids) != 1 or len(group_indices) != 1:
                raise RuntimeError("rollout group metadata is inconsistent")
            sample_id, group_index = next(iter(sample_ids)), next(iter(group_indices))
            teacher_row = bank.choose(sample_id, group_index)
            if teacher_row is None:
                continue
            bank_hits += 1
            try:
                teacher = build_teacher_sample(teacher_row, tokenizer)
            except (TeacherTrajectoryTooLongError, KeyError, RuntimeError, ValueError) as exc:
                skipped.append({"sample_id": sample_id, "group_index": group_index, "reason": str(exc)})
                continue

            # Rewards/advantages are already fixed. The inserted row is CE-only.
            slot = max(range(start, end), key=lambda index: len(data["tokens"][index]))
            data["tokens"][slot] = teacher["tokens"]
            data["response_lengths"][slot] = teacher["response_length"]
            data["loss_masks"][slot] = teacher["loss_mask"]
            data["teacher_log_probs"][slot] = teacher["teacher_marker"]
            data["truncated"][slot] = 0
            if "rollout_log_probs" in data:
                data["rollout_log_probs"][slot] = [0.0] * teacher["response_length"]
            replacements.append({
                "sample_id": sample_id,
                "group_index": group_index,
                "slot": slot,
                "teacher_trajectory_id": teacher["trajectory_id"],
                "teacher_ask_count": teacher["ask_count"],
                "teacher_target_tokens": teacher["target_token_count"],
            })

    mask_sums = [sum(mask) for mask in data["loss_masks"]]
    totals: dict[Any, int] = {}
    for rollout_id, mask_sum in zip(rollout_ids, mask_sums, strict=True):
        totals[rollout_id] = totals.get(rollout_id, 0) + mask_sum
    data["rollout_mask_sums"] = [totals[rollout_id] for rollout_id in rollout_ids]

    _write_audit({
        "protocol": "verified_teacher_supervision_v1",
        "enabled": _enabled("ENABLE_TEACHER_TRAJECTORIES"),
        "samples": len(flat),
        "groups": len(flat) // group_size,
        "all_negative_groups": all_negative_groups,
        "teacher_bank_hits": bank_hits,
        "teacher_replacements": len(replacements),
        "teacher_skips": skipped,
        "replacement_details": replacements,
    })
    return data
