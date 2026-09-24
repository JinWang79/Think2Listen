"""Joint GRPO, reference-KL, and optional verified-teacher objective."""

from __future__ import annotations

import os

import torch
from megatron.core import mpu

from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses
from slime.utils.ppo_utils import compute_approx_kl, compute_entropy_from_logits, compute_policy_loss


def _teacher_only_entropy(args, batch, logits, teacher_masks):
    """Return response-aligned entropy while materializing vocab work only for CE tokens."""
    response_entropy = []
    for (logits_chunk, _), teacher_mask in zip(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        ),
        teacher_masks,
        strict=True,
    ):
        active = teacher_mask.to(device=logits_chunk.device, dtype=torch.bool)
        if active.any():
            selected_entropy = compute_entropy_from_logits(
                logits_chunk[active].contiguous(),
                mpu.get_tensor_model_parallel_group(),
            )
            # Keep the existing per-sample reduction semantics without creating
            # another response-by-vocabulary tensor for non-teacher positions.
            response_entropy.append(logits_chunk.new_zeros(logits_chunk.size(0)).masked_scatter(active, selected_entropy))
        else:
            response_entropy.append(logits_chunk.new_zeros(logits_chunk.size(0)))
    return torch.cat(response_entropy, dim=0)


def joint_policy_teacher_loss(args, batch, logits, _sum_of_sample_mean):
    if args.advantage_estimator != "grpo":
        raise RuntimeError("Teacher CE v1 supports GRPO only")
    if args.use_opsm or args.use_tis or args.get_mismatch_metrics:
        raise RuntimeError("Teacher CE v1 does not yet support OPSM/TIS/mismatch filtering")
    if args.calculate_per_token_loss:
        raise RuntimeError("Teacher CE v1 expects sample-mean loss reduction")
    if mpu.get_context_parallel_world_size() != 1:
        raise RuntimeError("Teacher CE v1 has only been validated with context_parallel_size=1")
    if float(args.rollout_temperature) != 1.0:
        raise RuntimeError("Teacher CE v1 requires rollout_temperature=1.0 for an unscaled CE objective")
    if batch.get("teacher_log_probs") is None:
        raise RuntimeError("Teacher sample markers are missing from the mixed batch")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens")
    teacher_enabled = os.environ.get("ENABLE_TEACHER_TRAJECTORIES", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    teacher_entropy_coef = float(os.environ.get("TEACHER_SUPERVISION_ENTROPY_COEF", "0"))
    if teacher_entropy_coef < 0.0:
        raise RuntimeError("TEACHER_SUPERVISION_ENTROPY_COEF must be non-negative")
    need_rl_entropy = float(args.entropy_coef) != 0.0
    _, values = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_rl_entropy,
        max_seq_lens=max_seq_lens,
    )
    log_probs = torch.cat(values["log_probs"], dim=0)
    entropy = torch.cat(values["entropy"], dim=0) if need_rl_entropy else None
    old = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    old_log_probs = torch.cat(old, dim=0)
    advantages = torch.cat(batch["advantages"], dim=0)

    teacher_flags = [
        teacher_enabled and bool(marker.numel() and marker.max().item() > 0.5)
        for marker in batch["teacher_log_probs"]
    ]
    rl_masks = [mask if not flag else torch.zeros_like(mask) for mask, flag in zip(batch["loss_masks"], teacher_flags, strict=True)]
    teacher_masks = [mask if flag else torch.zeros_like(mask) for mask, flag in zip(batch["loss_masks"], teacher_flags, strict=True)]
    rl_reduce = get_sum_of_sample_mean(total_lengths, response_lengths, rl_masks, False, args.qkv_format, max_seq_lens)
    teacher_reduce = get_sum_of_sample_mean(total_lengths, response_lengths, teacher_masks, False, args.qkv_format, max_seq_lens)

    ppo_kl = old_log_probs - log_probs
    eps_clip_high = args.eps_clip if args.eps_clip_high is None else args.eps_clip_high
    pg_token_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, eps_clip_high)
    pg_loss = rl_reduce(pg_token_loss)
    entropy_loss = rl_reduce(entropy) if entropy is not None else 0.0 * log_probs.sum()
    ppo_kl_metric = rl_reduce(ppo_kl)
    loss = pg_loss - args.entropy_coef * entropy_loss

    kl_loss = None
    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type)
        kl_loss = rl_reduce(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    teacher_count = sum(teacher_flags)
    if teacher_count:
        teacher_ce = teacher_reduce(-log_probs)
        if teacher_entropy_coef:
            teacher_entropy = teacher_reduce(_teacher_only_entropy(args, batch, logits, teacher_masks))
        else:
            teacher_entropy = 0.0 * logits.sum()
    else:
        teacher_ce = 0.0 * logits.sum()
        teacher_entropy = 0.0 * logits.sum()
    teacher_lambda = (
        float(os.environ.get("TEACHER_SUPERVISION_LAMBDA", "0.1"))
        if teacher_enabled else 0.0
    )
    loss = loss + teacher_lambda * teacher_ce - teacher_entropy_coef * teacher_entropy

    reported = {
        "loss": loss.detach().clone(),
        "pg_loss": pg_loss.detach().clone(),
        "teacher_ce_loss": teacher_ce.detach().clone(),
        "teacher_entropy": teacher_entropy.detach().clone(),
        "teacher_entropy_coef": torch.tensor(teacher_entropy_coef, device=logits.device),
        "teacher_ce_weighted": (teacher_lambda * teacher_ce).detach().clone(),
        "teacher_entropy_weighted": (-teacher_entropy_coef * teacher_entropy).detach().clone(),
        "teacher_samples": torch.tensor(float(teacher_count), device=logits.device),
        "teacher_target_tokens": torch.stack([mask.sum() for mask in teacher_masks]).sum().float().detach(),
        "entropy_loss": entropy_loss.detach().clone(),
        "pg_clipfrac": rl_reduce(pg_clipfrac).detach().clone(),
        "ppo_kl": ppo_kl_metric.detach().clone(),
    }
    if kl_loss is not None:
        reported["kl_loss"] = kl_loss.detach().clone()
    return loss, reported


# Backward-compatible callable name for archived launcher metadata.
mixed_grpo_teacher_ce_loss = joint_policy_teacher_loss
