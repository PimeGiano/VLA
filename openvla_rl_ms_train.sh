#!/bin/bash
set -euo pipefail

VLA_PATH="./weights/openvla-7b-rlvla-warmup"
VLA_UNNORM_KEY="bridge_orig"
VLA_LOAD_PATH="" # lora checkpoint
SEED=42
STEPS_MAX=2000000
GRAD_ACCUM=20
ENV_ID="PutOnPlateInScene25Main-v3"

cuda=${1:-}
if [[ -z "${cuda}" ]]; then
  echo "Usage: $0 <comma-separated GPU ids>"
  exit 1
fi

IFS=',' read -ra gpus <<< "$cuda"
num_gpus=${#gpus[@]}

TIMESTAMP=$(date +"%m%d_%H%M")

EXTRA_ARGS=(
  --no-render-video
  --no-wandb
)

type=${2:-3}  # 1: RL4VLA, 2: ori, 3: dist

if [[ "${type}" -eq 1 ]]; then
  NAME="RL4VLA_${TIMESTAMP}"
  EXTRA_ARGS+=(
    --no-use_endoRM
    --no-update_valid_envs
    --warmup_ratio 0
    --distribution_loss_coef 0.0
  )
elif [[ "${type}" -eq 2 ]]; then
  NAME="ori_${TIMESTAMP}"
  EXTRA_ARGS+=(
    --no-use_endoRM
    --update_valid_envs
    --warmup_ratio 0
    --distribution_loss_coef 0.0
  )
elif [[ "${type}" -eq 3 ]]; then
  NAME="as-oft_simplest1.0_fix0.3_tau-1_${TIMESTAMP}"
  EXTRA_ARGS+=(
    --no-use_endoRM
    --update_valid_envs
    --warmup_ratio 0
    --dist_loss_type simplest  # simplest / gaussian_shape / gaussian_kernel
    --distribution_loss_coef 1.0
    --dist_fixed_sigma 0.3
    --dist_min_sigma 0.2
    --dist_tau -1.0
    --dist_trunc_k 0.0
    --kernel_sigma 1.0
    --dist_constrained_dims 6
    --dist_max_chunk_num 1
    --dist_action_dim 7
  )
elif [[ "${type}" -eq 4 ]]; then
  NAME="reward_smoothing_${TIMESTAMP}"
  EXTRA_ARGS+=(
    --use_endoRM
    --update_valid_envs
    --warmup_ratio 0
    --dist_loss_type simplest  # simplest / gaussian_shape / gaussian_kernel
    --distribution_loss_coef 0.0
    --dist_fixed_sigma 0.3
    --dist_min_sigma 0.2
    --dist_tau -1.0
    --dist_trunc_k 0.0
    --kernel_sigma 1.0
    --dist_constrained_dims 6
    --dist_max_chunk_num 1
    --dist_action_dim 7
    --endo_reward_scale 250.0
  )
fi


# NCCL_BLOCKING_WAIT=1 \
# NCCL_ASYNC_ERROR_HANDLING=1 \

TORCH_NCCL_BLOCKING_WAIT=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES="${cuda}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node="${num_gpus}" ./core/train_ms3_ppo_endo.py mani-skill-args \
  --name="${NAME}" \
  --env_id="${ENV_ID}" \
  --vla_path="${VLA_PATH}" \
  --vla_unnorm_key="${VLA_UNNORM_KEY}" \
  --vla_load_path="${VLA_LOAD_PATH}" \
  --seed="${SEED}" \
  --alg_gradient_accum="${GRAD_ACCUM}" \
  --steps_max="${STEPS_MAX}" \
  "${EXTRA_ARGS[@]}"
