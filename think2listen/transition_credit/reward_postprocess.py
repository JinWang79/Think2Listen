"""Group-adaptive reward calibration for Transition-as-Credit."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _enabled() -> bool:
    return os.environ.get("ENABLE_TRANSITION_CREDIT", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _group_norm(values: torch.Tensor, valid: torch.Tensor, args) -> torch.Tensor:
    count = valid.sum(dim=-1, keepdim=True)
    mean = (values * valid).sum(dim=-1, keepdim=True) / count.clamp_min(1)
    centered = values - mean
    if (
        getattr(args, "advantage_estimator", "") in {"grpo", "gspo"}
        and getattr(args, "grpo_std_normalization", False)
    ):
        variance = (
            ((values - mean).square() * valid).sum(dim=-1, keepdim=True)
            / (count - 1).clamp_min(1)
        )
        centered = centered / variance.sqrt().clamp_min(1e-6)
    return torch.where(valid.bool(), centered, torch.zeros_like(centered))


def post_process(args, samples):
    """Return terminal rewards and group-normalized calibrated advantages."""

    flat = (
        [item for group in samples for item in group]
        if samples and isinstance(samples[0], list)
        else list(samples)
    )
    if _enabled() and getattr(args, "advantage_estimator", "") not in {"grpo", "gspo"}:
        raise ValueError("Transition-as-Credit requires GRPO or GSPO")

    terminal_rewards = [float(sample.reward) for sample in flat]
    group_size = int(getattr(args, "n_samples_per_prompt", 1))
    if group_size <= 0 or len(flat) % group_size:
        raise ValueError(
            f"Group normalization requires group_size={group_size} to divide {len(flat)} samples"
        )

    terminal = torch.tensor(terminal_rewards, dtype=torch.float32).view(-1, group_size)
    valid = torch.tensor(
        [not bool(getattr(sample, "remove_sample", False)) for sample in flat],
        dtype=torch.bool,
    ).view(-1, group_size)
    classes = [
        str((sample.metadata or {}).get("transition_credit_outcome_class") or "invalid")
        for sample in flat
    ]
    misleading = [
        bool((sample.metadata or {}).get("transition_credit_misleading", False))
        for sample in flat
    ]

    calibrated = terminal.clone()
    beta_values: list[float | None] = [None] * len(flat)
    activated_groups = [False] * terminal.shape[0]
    if _enabled():
        beta_min = float(os.environ.get("TRANSITION_CREDIT_BETA_MIN", "0.5"))
        tau = float(os.environ.get("TRANSITION_CREDIT_TAU", "0.5"))
        temperature = float(os.environ.get("TRANSITION_CREDIT_TEMPERATURE", "0.1"))
        if not 0.0 <= beta_min <= 1.0 or temperature <= 0.0:
            raise ValueError("Require 0 <= TRANSITION_CREDIT_BETA_MIN <= 1 and temperature > 0")

        for group in range(terminal.shape[0]):
            start, end = group * group_size, (group + 1) * group_size
            correct = [i for i in range(start, end) if classes[i] == "correct"]
            incorrect = [i for i in range(start, end) if classes[i] == "incorrect"]
            misleading_incorrect = [i for i in incorrect if misleading[i]]
            if not correct or not incorrect or not misleading_incorrect:
                continue
            activated_groups[group] = True
            ratio = len(misleading_incorrect) / len(incorrect)
            beta = beta_min + (1.0 - beta_min) * float(
                torch.sigmoid(torch.tensor((ratio - tau) / temperature))
            )
            reference = [
                i
                for i in range(start, end)
                if i not in misleading_incorrect and bool(valid.flatten()[i])
            ]
            if not reference:
                continue
            anchor = min(float(terminal.flatten()[reference].mean()), 0.0)
            routed_reward = -beta + (1.0 - beta) * anchor
            for index in misleading_incorrect:
                calibrated.flatten()[index] = routed_reward
                beta_values[index] = beta

    valid_float = valid.to(dtype=torch.float32)
    advantages = (
        _group_norm(calibrated, valid_float, args)
        if getattr(args, "rewards_normalization", True)
        else calibrated * valid_float
    )

    audit_path = os.environ.get("TRANSITION_CREDIT_GROUP_AUDIT_PATH", "").strip()
    if audit_path and _enabled():
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for index, sample in enumerate(flat):
                group, member = divmod(index, group_size)
                handle.write(json.dumps({
                    "sample_id": (sample.metadata or {}).get("sample_id"),
                    "sample_index": getattr(sample, "index", None),
                    "group_index": getattr(sample, "group_index", group),
                    "group_member_index": member,
                    "terminal_reward": terminal_rewards[index],
                    "outcome_class": classes[index],
                    "misleading": misleading[index],
                    "group_activated": activated_groups[group],
                    "beta": beta_values[index],
                    "calibrated_reward": float(calibrated.flatten()[index]),
                    "combined_advantage": float(advantages.flatten()[index]),
                }, ensure_ascii=True) + "\n")
    return terminal_rewards, advantages.flatten().tolist()
