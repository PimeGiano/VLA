import os
import sys
import numpy as np
import random
from typing import List, TypedDict, Tuple, Dict, Any, Union
from termcolor import cprint
import torch

# 修改说明：LiberoWrapper 已修改为返回 numpy 格式数据，与 libero CPU 环境保持一致
# 主函数 train_ms3_ppo.py 中使用 to_numpy() 和 to_tensor_device() 进行格式转换

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark, get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

from libero.libero.envs import SubprocVectorEnv
# from ppo.envs.venv import SubprocVectorEnv
# from ppo.utils.util import add_info_board
# from ppo.envs.base import BaseEnv, EnvOutput

from core.benchmark.libero.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
)
from core.benchmark.libero.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from core.utils.gpu_order_align import uuid_align


class LiberoWrapper:
    def __init__(self, all_args, unnorm_state, extra_seed=0, device_id=None):
        self.args = all_args
        self.unnorm_state = unnorm_state

        self.num_envs = self.args.num_envs
        self.valid_envs_id = list(range(self.num_envs))
        self.last_valid_envs_id = list(range(self.num_envs))  # 上一帧有效环境ID缓存，用于对齐ManiSkill行为
        self.update_valid_envs = self.args.update_valid_envs
        assert self.args.img_width == self.args.img_height, "img_width and img_height must be the same in libero"
        self.resolution = self.args.img_width
        self.seeds = [self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)]

        # Convert local device_id to EGL device index for LIBERO rendering
        self.device_id = self._get_egl_device_id(device_id)
        # 为了能够与EGL的对应上，必须把CUDA_DIV设置为全部gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        self.model_family = self.args.model_family  # openvla

        self.current_step = 0
        self.num_steps_wait = self.args.num_steps_wait
        self.rollout_steps = None

        # 记录每个环境的成功状态（使用 numpy.bool_，与ManiSkill保持一致的最小改动）
        self.success = np.zeros(self.num_envs, dtype=bool)

        # Initialize task suite info with unified structure
        self.task_suite_name = {"train": {}, "test": {}}
        self.candidate_task_ids = {"train": {}, "test": {}}
        self.task_suite = {"train": {}, "test": {}}
        self.tasks = {"train": {}, "test": {}}
        self.initial_states_list = {"train": {}, "test": {}}
        self.env_creators = {"train": [], "test": []}
        self.sampled_suite_names = {"train": [], "test": []}
        self.sampled_task_ids = {"train": [], "test": []}

        # get train task_suite info
        for suite, task_ids in self.args.task_suite_name_train.items():
            self.task_suite_name["train"][suite] = suite
            self.candidate_task_ids["train"][suite] = task_ids
            self.task_suite["train"][suite] = get_benchmark(suite)(0)  # 这个0影响的是任务的排布, default=0 --> [0,1,2,3,4,...,9]

            # Initialize tasks and initial states for this suite
            self.tasks["train"][suite] = {}
            self.initial_states_list["train"][suite] = {}
            for task_id in task_ids:
                self.tasks["train"][suite][task_id] = self.task_suite["train"][suite].get_task(task_id)
                self.initial_states_list["train"][suite][task_id] = self.task_suite["train"][suite].get_task_init_states(task_id)

        # get test task_suite info
        for suite, task_ids in self.args.task_suite_name_test.items():
            self.task_suite_name["test"][suite] = suite
            self.candidate_task_ids["test"][suite] = task_ids
            self.task_suite["test"][suite] = get_benchmark(suite)(0)

            # Initialize tasks and initial states for this suite
            self.tasks["test"][suite] = {}
            self.initial_states_list["test"][suite] = {}
            for task_id in task_ids:
                self.tasks["test"][suite][task_id] = self.task_suite["test"][suite].get_task(task_id)
                self.initial_states_list["test"][suite][task_id] = self.task_suite["test"][suite].get_task_init_states(task_id)

        # Use mixed sampling across all suites for training
        self.sampled_suite_names["train"], self.sampled_task_ids["train"] = self._get_mix_task_suite_sample("train")
        self.env_creators["train"] = self._get_env_config("train", self.sampled_suite_names["train"], self.sampled_task_ids["train"])
        # For mixed sampling, we don't have a single current suite name
        
        # Use mixed sampling across all suites for testing
        self.sampled_suite_names["test"], self.sampled_task_ids["test"] = self._get_mix_task_suite_sample("test")
        self.env_creators["test"] = self._get_env_config("test", self.sampled_suite_names["test"], self.sampled_task_ids["test"])

        # create the environment, default is train
        self.current_env_type = "train"
        # current_suite_name is already set above based on sampling method
        self.env = SubprocVectorEnv(self.env_creators["train"])
        self.env.seed(self.seeds)
        self.env.reset()

        # variables
        # 奖励差分缓存，保持为 numpy.float32，避免在 wrapper 内进行不必要的类型转换
        self.reward_old = np.zeros((self.args.num_envs, 1), dtype=np.float32)  # [B, 1]

        # constants
        bins = np.linspace(-1, 1, 256)  # TODO 这里其实应该和policy是直接关联 后面再整理
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

    def _get_egl_device_id(self, ddp_rank):
        """
        Convert DDP rank to EGL device index for LIBERO rendering.
        Uses the existing uuid_align functionality for accurate device identification.

        Args:
            ddp_rank: DDP rank (usually local_rank)

        Returns:
            EGL device index that LIBERO's render_gpu_device_id expects
        """
        import os

        if ddp_rank is None:
            return -1  # Use default GPU selection

        try:
            # Get CUDA_VISIBLE_DEVICES to determine target GPU
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)

            if cuda_visible_devices is None:
                # If CUDA_VISIBLE_DEVICES is not set, use ddp_rank directly
                target_nvidia_gpu = ddp_rank
            else:
                # Parse CUDA_VISIBLE_DEVICES to get the target GPU
                visible_gpus = [int(gpu.strip()) for gpu in cuda_visible_devices.split(',')]
                if ddp_rank >= len(visible_gpus):
                    print(f"[LiberoWrapper] Warning: DDP rank {ddp_rank} >= number of visible GPUs {len(visible_gpus)}")
                    return -1
                target_nvidia_gpu = visible_gpus[ddp_rank]

            print(f"[LiberoWrapper] DDP rank {ddp_rank} -> Target NVIDIA GPU {target_nvidia_gpu}")

            # Use existing uuid_align functionality
            correspond_cuda2egl = uuid_align()

            # Get EGL device ID for the target NVIDIA GPU
            if target_nvidia_gpu in correspond_cuda2egl:
                egl_device_id = correspond_cuda2egl[target_nvidia_gpu]
                cprint(f"[LiberoWrapper] NVIDIA GPU {target_nvidia_gpu} -> EGL device {egl_device_id}", "green")
                return egl_device_id
            else:
                cprint(f"[LiberoWrapper] Warning: No EGL mapping found for NVIDIA GPU {target_nvidia_gpu}", "red")
                return target_nvidia_gpu  # Fallback to direct mapping

        except Exception as e:
            print(f"[LiberoWrapper] Error in GPU-EGL mapping: {e}")
            print(f"[LiberoWrapper] Falling back to DDP rank {ddp_rank} as EGL device ID")
            return ddp_rank

    def _get_mix_task_suite_sample(self, env_type: str = "train") -> Tuple[List[str], List[int]]:
        """
        Evenly distribute environments across all available suites and tasks.

        Args:
            env_type: Environment type ("train" or "test")

        Returns:
            Tuple of (suite_names, task_ids) where each list has length num_envs
        """
        # Get all available suites and their task IDs
        all_suite_task_pairs = []
        for suite_name, task_ids in self.candidate_task_ids[env_type].items():
            for task_id in task_ids:
                all_suite_task_pairs.append((suite_name, task_id))

        if not all_suite_task_pairs:
            raise ValueError(f"No tasks available for env_type: {env_type}")

        # Calculate how many environments each task should get
        total_tasks = len(all_suite_task_pairs)
        base_envs_per_task = self.num_envs // total_tasks
        extra_envs = self.num_envs % total_tasks

        # Distribute environments
        suite_names = []
        task_ids = []

        for i, (suite_name, task_id) in enumerate(all_suite_task_pairs):
            # Each task gets base_envs_per_task environments
            # The first extra_envs tasks get one additional environment
            num_envs_for_this_task = base_envs_per_task + (1 if i < extra_envs else 0)

            # Add this task's environments
            for _ in range(num_envs_for_this_task):
                suite_names.append(suite_name)
                task_ids.append(task_id)

        # Shuffle to randomize the order
        combined = list(zip(suite_names, task_ids))
        np.random.shuffle(combined)
        suite_names, task_ids = zip(*combined)

        return list(suite_names), list(task_ids)
    
    def _get_env_config(self, env_type: str, suite_names: Union[str, List[str]], task_ids: Union[np.ndarray, List[int]]) -> List:
        """
        Generate environment configuration for given task IDs.

        Args:
            env_type: Environment type ("train" or "test")
            suite_names: Task suite name(s) - can be a single string or list of strings
            task_ids: Array/List of task IDs to create environments for

        Returns:
            List of environment creator functions
        """
        env_creators = []

        # Handle both single suite and mixed suite cases
        if isinstance(suite_names, str):
            # Single suite case (backward compatibility)
            suite_names = [suite_names] * len(task_ids)

        if len(suite_names) != len(task_ids):
            raise ValueError(f"Length of suite_names ({len(suite_names)}) must match length of task_ids ({len(task_ids)})")

        for suite_name, task_id in zip(suite_names, task_ids):
            # Get task info
            if suite_name not in self.tasks[env_type]:
                raise ValueError(f"Unknown suite_name: {suite_name} for env_type={env_type}")
            if task_id not in self.tasks[env_type][suite_name]:
                raise ValueError(f"Unknown task_id: {task_id} for suite_name={suite_name}")
            task = self.tasks[env_type][suite_name][task_id]
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            # 校验BDDL路径存在性
            if not os.path.exists(task_bddl_file):
                raise FileNotFoundError(f"BDDL file not found: {task_bddl_file}")
            env_config = dict(
                bddl_file_name=task_bddl_file,
                camera_heights=self.resolution,
                camera_widths=self.resolution,
                render_gpu_device_id=self.device_id,
                reward_shaping=True,
                ignore_done=True
            )
            # 创建一个函数对象：lambda config=env_config: OffScreenRenderEnv(**config) 是一个匿名函数
            # 这一步不会创建环境, 把这个函数对象添加到 env_creators 列表中
            env_creators.append(lambda config=env_config: OffScreenRenderEnv(**config))

        return env_creators

    @property
    def episode_len(self):
        if self.rollout_steps is None:
            self.rollout_steps = {}
            # for train
            max_episode_len = 0
            for suite_name in self.task_suite_name["train"].keys():
                max_episode_len = max(max_episode_len, get_episode_len(suite_name))
            self.rollout_steps["train"] = max_episode_len
            # for test
            max_episode_len = 0
            for suite_name in self.task_suite_name["test"].keys():
                max_episode_len = max(max_episode_len, get_episode_len(suite_name))
            self.rollout_steps["test"] = max_episode_len

        return self.rollout_steps[self.current_env_type]

    def reset(self, env_type: str = 'train', same_init: bool = False, seed=None):
        # 对齐ManiSkill行为：增加参数与长度校验
        if env_type not in ("train", "test"):
            raise ValueError(f"env_type must be 'train' or 'test', got: {env_type}")
        # Switch environment if needed
        if env_type != self.current_env_type:
            self._switch_task_suite_set(env_type)

        self.current_step = 0
        # Reset success status for all environments（torch.bool）
        self.success = np.zeros(self.num_envs, dtype=bool)
        self.valid_envs_id = list(range(self.num_envs))
        self.last_valid_envs_id = list(range(self.num_envs))
        initial_states_to_set = []

        # 若提供评估种子，则在每个 env 的基础种子 self.seeds 上叠加该偏移，并立即 reset
        if seed is not None:
            assert isinstance(seed, int), "seed must be an integer"
            new_seeds = [base + seed for base in self.seeds]
            self.env.seed(new_seeds)
            self.env.reset()

        # 基本长度校验：抽样的 suite 与 task 数量需与 num_envs 一致
        suite_names = self.sampled_suite_names[self.current_env_type]
        task_ids = self.sampled_task_ids[self.current_env_type]
        if len(suite_names) != self.num_envs or len(task_ids) != self.num_envs:
            raise ValueError(f"Sampled suite/task length must equal num_envs. Got suites={len(suite_names)}, tasks={len(task_ids)}, num_envs={self.num_envs}")

        if same_init:
            # TODO: 评测同一初始状态（参考 openvla 逻辑），后续按需求完善
            pass
        else:
            # 随机采样初始状态
            for suite_name, task_id in zip(suite_names, task_ids):
                init_states = self.initial_states_list[self.current_env_type][suite_name][task_id]
                if len(init_states) == 0:
                    raise ValueError(f"No init states for suite={suite_name}, task_id={task_id}")
                idx = np.random.randint(len(init_states))
                initial_states_to_set.append(init_states[idx])

        obs = self.env.set_init_state(initial_states_to_set)

        # 获取每个环境的语言指令
        instruction = [
            self.tasks[self.current_env_type][suite_name][task_id].language
            for suite_name, task_id in zip(suite_names, task_ids)
        ]

        # 等待稳定化（dummy步）
        for _ in range(self.num_steps_wait):
            dummy_action = get_libero_dummy_action(self.model_family)
            dummy_actions = [dummy_action for _ in range(self.num_envs)]
            obs, _, _, _ = self.env.step(np.array(dummy_actions))

        obs_image_np = [get_libero_image(obs, self.resolution) for obs in obs]
        # 返回 numpy.uint8，避免不必要的类型转换
        obs_image = np.asarray(obs_image_np, dtype=np.uint8)

        # 重置奖励缓存（numpy）
        self.reward_old = np.zeros((self.num_envs, 1), dtype=np.float32)

        return obs_image, instruction, {}

    def step(self, raw_action):
        # 选择性推理：仅对未完成环境推理，其余填充dummy
        action = self._process_action_selective(raw_action)

        # 仅为有效环境提取动作
        valid_actions = [action[env_id] for env_id in self.valid_envs_id]

        # 当所有环境均已成功时，跳过环境推进，直接返回零增量与当前success
        if len(self.valid_envs_id) == 0:
            H, W = self.resolution, self.resolution
            obs_image = np.zeros((self.num_envs, H, W, 3), dtype=np.uint8)
            reward = np.zeros((self.num_envs, 1), dtype=np.float32)
            done = np.asarray(self.success, dtype=bool).reshape(-1, 1)
            info = {
                "step": [self.current_step for _ in range(self.num_envs)],
                "success": np.asarray(self.success).astype(bool),
                "task_suite_name": self.sampled_suite_names[self.current_env_type],
                "task_id": [str(x) for x in self.sampled_task_ids[self.current_env_type]],
            }
            self.current_step = min(self.current_step + 1, self.episode_len)
            if self.current_step == self.episode_len:
                info["episode"] = {}
                info["episode"]["success"] = info["success"]
                result = self._get_success_stats(info["success"])
                info["episode"].update(result)
            return obs_image, reward, done, info

        # 正常推进环境，加入异常上下文
        try:
            obs, _, done_valid, _ = self.env.step(np.array(valid_actions), self.valid_envs_id)
        except Exception as e:
            ctx = {
                "valid_envs_id": self.valid_envs_id,
                "current_step": self.current_step,
                "env_type": self.current_env_type,
                "sampled_suite_names": self.sampled_suite_names[self.current_env_type],
                "sampled_task_ids": self.sampled_task_ids[self.current_env_type],
            }
            raise RuntimeError(f"env.step failed with context: {ctx}") from e

        # 提取图像并转换为 torch.uint8
        obs_image_list = [get_libero_image(o, self.resolution) for o in obs]

        # 更新success（使用 numpy.bool_ 存储）
        for i, env_id in enumerate(self.valid_envs_id):
            if (not self.success[env_id]) and bool(done_valid[i]):
                pass
            self.success[env_id] = bool(done_valid[i]) or self.success[env_id]

        # 仅针对有效环境计算奖励（torch）
        reward_valid = self.get_reward(done_valid)  # [num_valid,1]

        # 回填至完整大小（numpy，与 libero CPU 环境保持一致）
        H, W = obs_image_list[0].shape[0], obs_image_list[0].shape[1]
        full_obs_image = np.zeros((self.num_envs, H, W, 3), dtype=np.uint8)
        full_reward = np.zeros((self.num_envs, 1), dtype=np.float32)

        if self.update_valid_envs:
            full_done = self.success.reshape(-1, 1)
        else:
            full_done = np.zeros((self.num_envs, 1), dtype=bool)

        for i, valid_env_id in enumerate(self.valid_envs_id):
            full_obs_image[valid_env_id] = obs_image_list[i]
            full_reward[valid_env_id] = reward_valid[i]

        # 组装info（success 使用 numpy.bool_ 格式，便于上层统一）
        info = {
            "step": [self.current_step for _ in range(self.num_envs)],
            "success": np.asarray(self.success).astype(bool),
            "task_suite_name": self.sampled_suite_names[self.current_env_type],
            "task_id": [str(x) for x in self.sampled_task_ids[self.current_env_type]],
        }
        self.current_step += 1
        if self.current_step == self.episode_len:
            info["episode"] = {}
            info["episode"]["success"] = info["success"]
            result = self._get_success_stats(info["success"])
            info["episode"].update(result)
            full_done = np.ones((self.num_envs, 1), dtype=bool)  # truncate all at episode end

        return full_obs_image, full_reward, full_done, info

    def get_reward(self, done_valid_envs):
        """
        计算奖励：输入为有效环境的done标记（布尔），返回对应的奖励变化量。
        输出为 numpy.float32 [num_valid, 1]
        """
        current_reward = np.asarray(done_valid_envs, dtype=np.float32).reshape(-1, 1)
        # 从缓存中提取对应有效环境的旧奖励
        old_reward_valid = np.zeros_like(current_reward)
        for i, env_id in enumerate(self.valid_envs_id):
            old_reward_valid[i] = self.reward_old[env_id]
        # 计算差分
        reward_diff = current_reward - old_reward_valid
        # 回写缓存
        for i, env_id in enumerate(self.valid_envs_id):
            self.reward_old[env_id] = current_reward[i]
        return reward_diff

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        # TODO 这个函数应该是policy和环境都相关的 应该拆分，提取action以及Unnormalize分属于两部分

        # Extract predicted action tokens and translate into (normalized) continuous actions
        # this part is related to openvla;
        # same as 'def decode_token_ids_to_actions' in openvla/prismatic/vla/action_tokenizer.py
        pact_token = raw_actions.cpu().numpy()  # [B, dim]
        dact = 32000 - pact_token  # [B, dim]
        dact = np.clip(dact - 1, a_min=0, a_max=254)  # [B, dim]
        normalized_actions = np.asarray([self.bin_centers[da] for da in dact])  # [B, dim]

        # TODO 参考vlarl 可以添加类似于格式惩罚 invalid_mask = (action == -100.0).any(axis=1) 但是predict_action_batch已经做了assert 再看看怎么处理

        # Unnormalize actions
        # this part is related to simulator
        action_norm_stats = self.unnorm_state
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))  # [dim]
        mask = np.asarray(mask).reshape(1, -1)  # [1, dim]
        action_high = np.array(action_norm_stats["q99"]).reshape(1, -1)  # [1, dim]
        action_low = np.array(action_norm_stats["q01"]).reshape(1, -1)  # [1, dim]
        raw_action_np = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        # 参考openval run_libero_eval.py#L219
        action = normalize_gripper_action(raw_action_np, binarize=True)
        if self.model_family == "openvla":
            action = invert_gripper_action(action)

        return action

    def _process_action_selective(self, raw_actions: torch.Tensor) -> np.ndarray:
        """
        仅对未完成环境进行推理；对已完成环境使用dummy动作，保持 batch 对齐。
        对齐ManiSkill：当 valid_envs 不是全体时，要求 raw_actions 的 batch 与 len(valid_envs_id) 匹配。
        """
        # Initialize all actions with dummy actions
        if len(self.valid_envs_id) == self.num_envs:
            step_actions = self._process_action(raw_actions)
        else:
            # Initialize all actions with dummy actions
            dummy_action = get_libero_dummy_action(self.model_family)
            step_actions = [dummy_action for _ in range(self.num_envs)]

            if len(self.valid_envs_id) > 0:
                predicted_step_action = self._process_action(raw_actions)

                # Update actions for incomplete environments
                for idx in range(min(len(self.valid_envs_id), predicted_step_action.shape[0])):
                    env_id = self.valid_envs_id[idx]
                    if not bool(self.success[env_id]):
                        step_actions[env_id] = predicted_step_action[idx]

        return np.array(step_actions)

    def _switch_task_suite_set(self, env_type: str, resample: bool = False):
        """Switch to a different task suite or environment type."""
        # Close current environment
        self.close()

        # TODO 如果环境数量小的话应该每次重新抽样重构环境，才能比较完整的测试，不过多卡也能一定程度上保证完整
        if resample:
            self.sampled_suite_names[self.current_env_type], self.sampled_task_ids[self.current_env_type] = self._get_mix_task_suite_sample(env_type)
            self.env_creators[env_type] = self._get_env_config(env_type, self.sampled_suite_names, self.sampled_task_ids)

        # Create new environment
        self.env = SubprocVectorEnv(self.env_creators[env_type])
        self.env.seed(self.seeds)
        self.env.reset()

        # Update current state
        self.current_env_type = env_type

    def close(self):
        """Close the environment."""
        self.env.close()

    def _get_success_stats(self, success_list):
        """Get success statistics for each task suite and task ID."""
        suite_names = self.sampled_suite_names[self.current_env_type]
        task_ids = self.sampled_task_ids[self.current_env_type]

        # 统计每个suite的任务数量，每个suite的成功率；每个suite下每个taskid的任务数量，及成功率
        success_stats = {}
        for suite_name, task_id, success in zip(suite_names, task_ids, success_list):
            # 兼容 torch.bool / numpy.bool_ / python bool
            try:
                success_val = bool(success)
            except Exception:
                success_val = True if str(success).lower() == "true" else False
            suite_key = f"{suite_name}_s"
            task_key = f"{suite_name}_{task_id}_s"
            if suite_key not in success_stats:
                success_stats[suite_key] = []
            success_stats[suite_key] += [success_val]
            if task_key not in success_stats:
                success_stats[task_key] = []
            success_stats[task_key] += [success_val]

        return success_stats
    
    def update_valid_envs_id(self):
        """Update the list of valid environment IDs."""
        if not self.update_valid_envs:
            return
        # 缓存上一次有效环境ID，并更新当前有效环境ID列表
        self.last_valid_envs_id = self.valid_envs_id
        self.valid_envs_id = [env_id for env_id, d in enumerate(self.success) if not bool(d)]

    def update_endogenous_reward(self, env_reward, endo_reward, valid_envs_id=None):
        """
        Update reward by combining environment reward with endogenous reward.

        Args:
            env_reward: Environment reward array [num_envs, 1] (numpy or torch)
            endo_reward: Endogenous reward tensor [num_valid_envs, 1] or None if use_endoRM=False

        Returns:
            Combined reward array [num_envs, 1] (same type as env_reward)
        """
        if not self.args.use_endoRM:
            # When endogenous reward is disabled, return only environment reward
            return env_reward
        if valid_envs_id is None:
            valid_envs_id = self.valid_envs_id

        # Convert inputs to numpy for consistent processing (libero runs on CPU)
        from core.utils.utils import to_numpy
        env_reward_np = to_numpy(env_reward)
        endo_reward_np = to_numpy(endo_reward) if endo_reward is not None else None

        # When endogenous reward is enabled, combine both rewards
        # endo_reward only contains rewards for valid environments
        reward_array = np.copy(env_reward_np)

        if endo_reward_np is not None:
            assert len(valid_envs_id) == endo_reward_np.shape[0]
            assert endo_reward_np.shape[1] == env_reward_np.shape[1]

            # Add endogenous rewards for valid environments
            for i, env_id in enumerate(valid_envs_id):
                if i < len(endo_reward_np):  # Safety check
                    reward_array[env_id] += endo_reward_np[i]

        return reward_array


def get_episode_len(task_suite_name) -> int:
    """
    Determine max_step dynamically based on the task suite.

    Returns:
        int: Maximum number of steps allowed for the task.
    """
    task_max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    task_max_step = task_max_steps.get(task_suite_name, 300)  # Default to 300 if not specified
    return task_max_step
