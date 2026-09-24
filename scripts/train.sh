#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${SLIME_DIR:?Set SLIME_DIR to the pinned SLIME checkout}"
: "${MEGATRON_DIR:?Set MEGATRON_DIR to the compatible Megatron-LM checkout}"
: "${SLIME_MODEL_CONFIG:?Set SLIME_MODEL_CONFIG to a SLIME model-argument script}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Hugging Face-format actor checkpoint}"
: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the distributed training checkpoint}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the training JSONL file}"
: "${EVAL_CONFIG:?Set EVAL_CONFIG to the evaluation YAML file}"
: "${AUDIO_MODEL_NAME_OR_PATH:?Set AUDIO_MODEL_NAME_OR_PATH}"

ENABLE_TRANSITION_CREDIT="${ENABLE_TRANSITION_CREDIT:-1}"
ENABLE_TEACHER_TRAJECTORIES="${ENABLE_TEACHER_TRAJECTORIES:-1}"
if [[ "${ENABLE_TRANSITION_CREDIT}" == "1" ]]; then
  : "${TRANSITION_CREDIT_EXTRACTOR_MODEL:?Required when Transition-as-Credit is enabled}"
fi
if [[ "${ENABLE_TEACHER_TRAJECTORIES}" == "1" ]]; then
  : "${TEACHER_TRAJECTORY_BANK_PATH:?Required when teacher trajectories are enabled}"
  : "${TEACHER_TRAJECTORY_TOKENIZER_PATH:?Required when teacher trajectories are enabled}"
fi

RUN_NAME="${RUN_NAME:-think2listen-review}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/artifacts/runs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/artifacts/checkpoints}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SAVE_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"

if [[ -e "${RUN_DIR}" || -e "${SAVE_DIR}" ]]; then
  echo "Refusing to overwrite an existing run or checkpoint directory" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}" "${SAVE_DIR}"

export ENABLE_TRANSITION_CREDIT ENABLE_TEACHER_TRAJECTORIES
export AUDIO_MODEL_BASE_URL="${AUDIO_MODEL_BASE_URL:-http://127.0.0.1:8901/v1}"
export AUDIO_PROMPT_VERSION="${AUDIO_PROMPT_VERSION:-targeted-audio-v1}"
export AUDIO_CACHE_PATH="${AUDIO_CACHE_PATH:-${RUN_DIR}/audio_observations.sqlite}"
export TRANSITION_CREDIT_EXTRACTOR_BASE_URL="${TRANSITION_CREDIT_EXTRACTOR_BASE_URL:-http://127.0.0.1:8006/v1}"
export TRANSITION_CREDIT_BETA_MIN="${TRANSITION_CREDIT_BETA_MIN:-0.5}"
export TRANSITION_CREDIT_TAU="${TRANSITION_CREDIT_TAU:-0.5}"
export TRANSITION_CREDIT_TEMPERATURE="${TRANSITION_CREDIT_TEMPERATURE:-0.1}"
export TRANSITION_CREDIT_EXTRACTOR_AUDIT_PATH="${TRANSITION_CREDIT_EXTRACTOR_AUDIT_PATH:-${RUN_DIR}/transition_extractor.jsonl}"
export TRANSITION_CREDIT_GROUP_AUDIT_PATH="${TRANSITION_CREDIT_GROUP_AUDIT_PATH:-${RUN_DIR}/transition_credit.jsonl}"
export TEACHER_TRAJECTORY_AUDIT_PATH="${TEACHER_TRAJECTORY_AUDIT_PATH:-${RUN_DIR}/teacher_trajectories.jsonl}"
export TEACHER_TRAJECTORY_MAX_TOTAL_TOKENS="${TEACHER_TRAJECTORY_MAX_TOTAL_TOKENS:-8192}"
export TEACHER_SUPERVISION_LAMBDA="${TEACHER_SUPERVISION_LAMBDA:-0.1}"
export TEACHER_SUPERVISION_ENTROPY_COEF="${TEACHER_SUPERVISION_ENTROPY_COEF:-0}"
export REWARD_AUDIT_PATH="${REWARD_AUDIT_PATH:-${RUN_DIR}/rewards.jsonl}"
export MIN_ASK_STEPS="${MIN_ASK_STEPS:-0}"
export MAX_ASK_STEPS="${MAX_ASK_STEPS:-4}"
export ASK_COST="${ASK_COST:-0}"
export ASK_BONUS="${ASK_BONUS:-0}"
export MULTI_ASK_ENV="${MULTI_ASK_ENV:-real}"
export ACTOR_ENABLE_THINKING="${ACTOR_ENABLE_THINKING:-1}"

source "${SLIME_MODEL_CONFIG}"
RUNTIME_ENV_JSON="$(python3 "${REPO_ROOT}/scripts/build_runtime_env.py")"
printf '%s\n' "${RUNTIME_ENV_JSON}" > "${RUN_DIR}/runtime_env.json"

TRAIN_COMMAND=(
  python3 "${SLIME_DIR}/train.py"
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --start-rollout-id 0 --colocate
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${BASE_CHECKPOINT}" --load "${BASE_CHECKPOINT}"
  --no-load-optim --no-load-rng --finetune
  --save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-20}"
  --no-save-optim --no-save-rng
  --prompt-data "${PROMPT_DATA}" --input-key prompt --label-key label
  --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking": true}'
  --eval-config "${EVAL_CONFIG}" --eval-interval "${EVAL_INTERVAL:-20}"
  --custom-generate-function-path think2listen.rollout.generator.custom_generate
  --custom-rm-path think2listen.training.reward.custom_rm
  --custom-reward-post-process-path think2listen.transition_credit.reward_postprocess.post_process
  --custom-convert-samples-to-train-data-path think2listen.training.data.convert_samples_to_train_data
  --loss-type custom_loss
  --custom-loss-function-path think2listen.training.loss.joint_policy_teacher_loss
  --num-epoch "${NUM_EPOCH:-5}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
  --rollout-temperature 1.0
  --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
  --advantage-estimator grpo
  --use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.001}" --kl-loss-type low_var_kl
  --entropy-coef 0.0 --eps-clip 0.2
  --optimizer adam --lr "${LEARNING_RATE:-1e-6}" --lr-decay-style constant
  --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.2}"
  --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash
  --save-debug-rollout-data "${RUN_DIR}/rollout-{rollout_id}.pt"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${RUN_DIR}/train_command.txt"
printf '\n' >> "${RUN_DIR}/train_command.txt"
ray job submit --address="${RAY_DASHBOARD_ADDRESS}" --runtime-env-json="${RUNTIME_ENV_JSON}" -- "${TRAIN_COMMAND[@]}"
