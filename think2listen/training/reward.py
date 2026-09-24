"""Terminal outcome reward with optional Transition-as-Credit extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path

from think2listen.rollout.cache import normalize_query
from think2listen.rollout.protocol import parse_answer, parse_inline_action


CORRECT_REWARD = 1.0
INCORRECT_REWARD = -1.0
DEFAULT_ASK_COST = 0.0
DEFAULT_ASK_BONUS = 0.0
INVALID_ACTION_PENALTY = -1.0
DUPLICATE_QUERY_PENALTY = -0.10
DEFAULT_LENGTH_PENALTY_FREE_TOKENS = 4096
DEFAULT_LENGTH_PENALTY_PER_1024 = 0.10
DEFAULT_LENGTH_PENALTY_MAX = 0.20


def _enabled(name: str, *, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


async def custom_rm(args, sample, **kwargs):
    metadata = sample.metadata or {}
    ask_cost = float(os.environ.get("ASK_COST", str(DEFAULT_ASK_COST)))
    ask_bonus = float(os.environ.get("ASK_BONUS", str(DEFAULT_ASK_BONUS)))
    trajectory = metadata.get("e2e_trajectory") or []
    final_action = str(metadata.get("e2e_final_action") or "")
    answer = parse_answer(final_action)
    if answer is None and metadata.get("e2e_actor_enable_thinking"):
        inline_action = parse_inline_action(final_action)
        if inline_action is not None and inline_action[0] == "answer":
            answer = inline_action[1]
    terminal_status = str(metadata.get("e2e_terminal_status") or "missing_answer")
    # ``loss_mask`` represents exactly the actor-generated tokens that can
    # receive policy-gradient loss.  It intentionally excludes prompt,
    # environment/template, and synthetic repair tokens, unlike
    # ``response_length``.
    actor_generated_token_count = int(sum(sample.loss_mask or []))
    length_penalty_free_tokens = int(os.environ.get(
        "LENGTH_PENALTY_FREE_TOKENS", str(DEFAULT_LENGTH_PENALTY_FREE_TOKENS)
    ))
    length_penalty_per_1024 = float(os.environ.get(
        "LENGTH_PENALTY_PER_1024", str(DEFAULT_LENGTH_PENALTY_PER_1024)
    ))
    length_penalty_max = float(os.environ.get(
        "LENGTH_PENALTY_MAX", str(DEFAULT_LENGTH_PENALTY_MAX)
    ))
    if length_penalty_free_tokens < 0 or length_penalty_per_1024 < 0 or length_penalty_max < 0:
        raise ValueError("length-penalty settings must be non-negative")
    queries = [str(item.get("query") or "") for item in trajectory]
    observations = [str(item.get("observation") or "") for item in trajectory]
    audit = {
        "sample_id": metadata.get("sample_id"), "group_index": sample.group_index, "sample_index": sample.index,
        "ask_count": len(trajectory), "queries": queries, "observations": observations,
        "terminal_status": terminal_status, "answer": answer, "correct_letter": metadata.get("correct_letter"),
        "environment_errors": list(metadata.get("e2e_environment_errors") or []),
    }
    for key in ("sweep_condition", "sweep_temperature", "sweep_n_samples"):
        if key in metadata:
            audit[key] = metadata[key]
    if terminal_status == "infrastructure_error":
        sample.remove_sample = True
        orm_reward, status = 0.0, "infrastructure_error"
    elif terminal_status != "answered" or answer is None or answer not in dict(metadata.get("choices") or {}):
        # A terminal ``invalid_action`` has generated a completion but could
        # not be parsed or conservatively repaired into an executable action.
        # It carries no reliable action label, so exclude it entirely from
        # GRPO normalization and policy-gradient loss rather than treating it
        # as an ordinary incorrect answer.  Length-capped invalid actions are
        # already removed in the generator; repeat the invariant here.
        if terminal_status in {"invalid_action", "invalid_action_response_max_tokens"}:
            sample.remove_sample = True
        orm_reward, status = INVALID_ACTION_PENALTY, "invalid_action"
    else:
        duplicates = len({normalize_query(query) for query in queries}) != len(queries)
        orm_reward = (
            (CORRECT_REWARD if answer == str(metadata["correct_letter"]) else INCORRECT_REWARD)
            + (ask_cost + ask_bonus) * len(queries)
        )
        if duplicates:
            orm_reward += DUPLICATE_QUERY_PENALTY
        status = "scored"

    # Penalize only trainable, protocol-valid terminal answers. Removed and
    # invalid/truncated samples must never receive a second, length-derived
    # reward signal.
    raw_orm_reward = orm_reward
    length_penalty = 0.0
    if status == "scored" and not sample.remove_sample:
        excess_actor_tokens = max(0, actor_generated_token_count - length_penalty_free_tokens)
        length_penalty = min(
            length_penalty_max,
            length_penalty_per_1024 * excess_actor_tokens / 1024.0,
        )
        orm_reward -= length_penalty
    outcome_class = (
        "correct" if status == "scored" and answer == str(metadata.get("correct_letter"))
        else "incorrect" if status == "scored" else "invalid"
    )
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["transition_credit_outcome_class"] = outcome_class

    # Transition-as-Credit only evaluates terminally incorrect, valid answers.
    # The group postprocessor later activates calibration only in mixed-outcome
    # groups, matching the method described in the paper.
    if _enabled("ENABLE_TRANSITION_CREDIT") and raw_orm_reward == INCORRECT_REWARD:
        from think2listen.transition_credit.runtime import score_runtime_sample

        try:
            transition = await score_runtime_sample(args, sample)
        except Exception as exc:
            status = "transition_credit_extractor_error"
            audit["transition_credit_error"] = repr(exc)
        else:
            audit["transition_credit"] = transition
            sample.metadata["transition_credit_runtime"] = transition
            sample.metadata["transition_credit_misleading"] = bool(
                transition.get("misleading", False)
            )

    audit.update({
        "status": status,
        "raw_orm_reward": raw_orm_reward,
        "orm_reward": orm_reward,
        "actor_generated_token_count": actor_generated_token_count,
        "length_penalty_free_tokens": length_penalty_free_tokens,
        "length_penalty_per_1024": length_penalty_per_1024,
        "length_penalty_max": length_penalty_max,
        "length_penalty": length_penalty,
        "ask_cost": ask_cost,
        "ask_bonus": ask_bonus,
    })

    # The post-process hook performs group-adaptive reward calibration.
    sample.reward = float(orm_reward)

    audit["reward"] = sample.reward
    path = os.environ.get("REWARD_AUDIT_PATH")
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit, ensure_ascii=True) + "\n")
    return sample.reward
