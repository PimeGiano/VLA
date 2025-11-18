#!/usr/bin/env bash
set -euo pipefail

# ManiSkill OOD 渲染脚本（OpenVLA 权重）
# usage: ./new_scripts/openvla_ms_ood_eval.sh [cuda] [eval_type|custom] [lora_override] [eval_seeds] [ckpt_override]
#   cuda           : CUDA_VISIBLE_DEVICES（默认 0）
#   eval_type      : warmup | rl | sft
#   lora_override  : 可选 LoRA / finetune checkpoint
#   eval_seeds     : 逗号分隔的渲染评估种子（默认 "42,1234"）
#   ckpt_override  : 需要覆盖默认模型路径时使用

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$( dirname "${SCRIPT_DIR}" )"

cuda="${1:-0}"
eval_type="${2:-rl}"
lora_override="${3:-}"
eval_seeds="${4:-42,1234}"
ckpt_override="${5:-}"

BASE_SEED="${BASE_SEED:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
VLA_UNNORM_KEY="bridge_orig"

case "${eval_type}" in
  warmup)
    CKPT_PATH="${ckpt_override:-./weights/openvla/openvla-7b}"
    VLA_LOAD_PATH="${lora_override:-}"
    ;;
  rl)
    CKPT_PATH="${ckpt_override:-./weights/openvla-7b-rlvla-warmup}"
    VLA_LOAD_PATH="${lora_override:-}"
    ;;
  sft)
    CKPT_PATH="${ckpt_override:-./weights/openvla-7b-rlvla-warmup}"
    VLA_LOAD_PATH="${lora_override:-}"
    VLA_UNNORM_KEY="sft"
    ;;
esac


timestamp="$(date +"%m%d%H%M%S")"

SUMMARY_FILE="$(mktemp)"

print_summary() {
  printf "\033[36m===== Progress Summary =====\033[0m\n"
  cat "${SUMMARY_FILE}"
  printf "\033[36m============================\033[0m\n"
}

run_suite() {
  local obj_set="$1"
  shift
  local env_ids=("$@")

  for env_id in "${env_ids[@]}"; do
    NAME="eval_${eval_type}_${env_id}_seed${BASE_SEED}_${timestamp}"

    local cmd=(
      python "${REPO_PATH}/core/eval_ms3_ppo.py" mani-skill-args
      --env_id "${env_id}"
      --env_type "${obj_set}"
      --vla_path "${CKPT_PATH}"
      --vla_unnorm_key "${VLA_UNNORM_KEY}"
      --vla_load_path "${VLA_LOAD_PATH}"
      --seed "${BASE_SEED}"
      --num_envs "${NUM_ENVS}"
      --eval_seeds "${eval_seeds}"
      --name "${NAME}"
      --only_render
      --no-wandb
    )

    local start_ts end_ts duration status
    start_ts=$(date +%s)
    if CUDA_VISIBLE_DEVICES="${cuda}" \
       XLA_PYTHON_CLIENT_PREALLOCATE=false \
       "${cmd[@]}"; then
      status="OK"
    else
      status="FAIL"
    fi
    end_ts=$(date +%s)
    duration=$(( end_ts - start_ts ))
    printf "[%s] %s (%s) finished in %s seconds.\n" "${status}" "${env_id}" "${obj_set}" "${duration}" | tee -a "${SUMMARY_FILE}"
    print_summary
  done
}

TEST_ENVS=(
  "PutOnPlateInScene25VisionImage-v1"
  "PutOnPlateInScene25VisionTexture03-v1"
  "PutOnPlateInScene25VisionTexture05-v1"
  "PutOnPlateInScene25VisionWhole03-v1"
  "PutOnPlateInScene25VisionWhole05-v1"

  "PutOnPlateInScene25Carrot-v1"
  "PutOnPlateInScene25Plate-v1"
  "PutOnPlateInScene25Instruct-v1"
  "PutOnPlateInScene25MultiCarrot-v1"
  "PutOnPlateInScene25MultiPlate-v1"

  "PutOnPlateInScene25Position-v1"
  "PutOnPlateInScene25EEPose-v1"
  "PutOnPlateInScene25PositionChangeTo-v1"
)

TRAIN_ENVS=(
  "PutOnPlateInScene25MultiCarrot-v1"
  "PutOnPlateInScene25MultiPlate-v1"
)

IND_ENVS=(
  "PutOnPlateInScene25Main-v3"
)
run_suite "train" "${IND_ENVS[@]}"
#run_suite "test" "${IND_ENVS[@]}"
run_suite "test" "${TEST_ENVS[@]}"
run_suite "train" "${TRAIN_ENVS[@]}"

echo "OpenVLA ManiSkill OOD render完成"
