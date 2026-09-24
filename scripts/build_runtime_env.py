#!/usr/bin/env python3
"""Build the Ray runtime environment from explicit, anonymous configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


repo_root = Path(__file__).resolve().parents[1]
slime_dir = required("SLIME_DIR")
megatron_dir = required("MEGATRON_DIR")
env_names = [
    "CUDA_VISIBLE_DEVICES",
    "AUDIO_MODEL_BASE_URL",
    "AUDIO_MODEL_API_KEY",
    "AUDIO_MODEL_NAME_OR_PATH",
    "AUDIO_PROMPT_VERSION",
    "AUDIO_CACHE_PATH",
    "ENABLE_TRANSITION_CREDIT",
    "TRANSITION_CREDIT_EXTRACTOR_BASE_URL",
    "TRANSITION_CREDIT_EXTRACTOR_API_KEY",
    "TRANSITION_CREDIT_EXTRACTOR_MODEL",
    "TRANSITION_CREDIT_EXTRACTOR_MAX_TOKENS",
    "TRANSITION_CREDIT_EXTRACTOR_TIMEOUT_SECONDS",
    "TRANSITION_CREDIT_EXTRACTOR_ATTEMPTS",
    "TRANSITION_CREDIT_EXTRACTOR_MAX_CONCURRENCY",
    "TRANSITION_CREDIT_BETA_MIN",
    "TRANSITION_CREDIT_TAU",
    "TRANSITION_CREDIT_TEMPERATURE",
    "TRANSITION_CREDIT_EXTRACTOR_AUDIT_PATH",
    "TRANSITION_CREDIT_GROUP_AUDIT_PATH",
    "ENABLE_TEACHER_TRAJECTORIES",
    "TEACHER_TRAJECTORY_BANK_PATH",
    "TEACHER_TRAJECTORY_TOKENIZER_PATH",
    "TEACHER_TRAJECTORY_REQUIRED_AUDIO_PROMPT_VERSION",
    "TEACHER_TRAJECTORY_MAX_TOTAL_TOKENS",
    "TEACHER_TRAJECTORY_BUCKET_WEIGHTS",
    "TEACHER_TRAJECTORY_AUDIT_PATH",
    "TEACHER_SUPERVISION_LAMBDA",
    "TEACHER_SUPERVISION_ENTROPY_COEF",
    "REWARD_AUDIT_PATH",
    "MIN_ASK_STEPS",
    "MAX_ASK_STEPS",
    "ASK_COST",
    "ASK_BONUS",
    "MULTI_ASK_ENV",
    "ACTOR_ENABLE_THINKING",
    "E2E_TOTAL_RESPONSE_TOKEN_BUDGET",
]
env_vars = {name: value for name in env_names if (value := os.environ.get(name)) is not None}
env_vars.update({
    "PYTHONPATH": os.pathsep.join((str(repo_root), slime_dir, megatron_dir)),
    "PYTHONSAFEPATH": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
})
print(json.dumps({"working_dir": slime_dir, "excludes": [".git"], "env_vars": env_vars}))
