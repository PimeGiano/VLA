import gymnasium as gym
import numpy as np
import torch
from collections import deque
from mani_skill.envs.sapien_env import BaseEnv


def get_maniskill_dummy_action():
    """Get dummy/no-op action for ManiSkill environments."""
    return [0, 0, 0, 0, 0, 0, -1]


class ManiSkillWrapper:
    def __init__(self, all_args, unnorm_state, extra_seed=0, device_id=None):
        self.args = all_args
        self.unnorm_state = unnorm_state

        self.num_envs = self.args.num_envs
        self.valid_envs_id = list(range(self.num_envs))
        self.last_valid_envs_id = list(range(self.num_envs))
        self.update_valid_envs = getattr(self.args, 'update_valid_envs', False)
        self.episode_len = self.args.episode_len

        # Track success status for each environment
        self.success = np.zeros(self.num_envs, dtype=bool)
        robot_control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"

        env_config = dict(
            id=self.args.env_id,
            num_envs=self.args.num_envs,
            obs_mode="rgb+segmentation",
            control_mode=robot_control_mode,
            sim_backend=f"cuda:{device_id}",
            render_backend=f"cuda:{device_id}",
            sim_config={
                "sim_freq": 500,
                "control_freq": 5,
            },
            max_episode_steps=self.args.episode_len,
            sensor_configs={"shader_pack": "default"},
        )
        self.env: BaseEnv = gym.make(**env_config)
        self.seeds = [self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)]
        self.env.reset(seed=self.seeds)

        # variables
        self.reward_old = torch.zeros(self.args.num_envs, 1, dtype=torch.float32)  # [B, 1]

        # constants
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

        # endogenous reward
        self.use_avg_endo = getattr(self.args, 'use_avg_endo', False)
        reward_window_size = getattr(self.args, 'reward_window_size', 10)
        self.endogenous_reward = [deque(maxlen=reward_window_size) for _ in range(self.num_envs)]

    def get_reward(self, info):
        reward = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(info["success"].device)  # [B, 1]

        # reward += info["is_src_obj_grasped"].reshape(-1, 1) * 0.1
        # reward += info["consecutive_grasp"].reshape(-1, 1) * 0.1
        reward += (info["success"].reshape(-1, 1) & info["is_src_obj_grasped"].reshape(-1, 1)) * 1.0

        # diff
        reward_diff = reward - self.reward_old
        self.reward_old = reward

        return reward_diff

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        # TODO 限制在 PyTorch 内部 使用张量操作而不转换为 NumPy 数组
        action_scale = 1.0

        # Extract predicted action tokens and translate into (normalized) continuous actions
        pact_token = raw_actions.cpu().numpy()  # [B, dim]
        dact = 32000 - pact_token  # [B, dim]
        dact = np.clip(dact - 1, a_min=0, a_max=254)  # [B, dim]
        normalized_actions = np.asarray([self.bin_centers[da] for da in dact])  # [B, dim]

        # Unnormalize actions
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

        raw_action = {
            "world_vector": raw_action_np[:, :3],
            "rotation_delta": raw_action_np[:, 3:6],
            "open_gripper": raw_action_np[:, 6:7],  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * action_scale  # [B, 3]
        action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0  # [B, 1]

        # origin euler
        action["rot_axangle"] = raw_action["rotation_delta"]

        action = {k: torch.tensor(v) for k, v in action.items()}  # to float32 ?

        action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)

        # to tpdv
        action = action.to(raw_actions.device)

        return action

    def reset(self, env_type: str, same_init: bool = False, seed=None):
        # Reset success status and valid environments
        self.success = np.zeros(self.num_envs, dtype=bool)
        self.valid_envs_id = list(range(self.num_envs))
        self.last_valid_envs_id = list(range(self.num_envs))

        options = {}
        # 具体的定义在ManiSkill/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene_multi.py
        options["obj_set"] = env_type
        if same_init:
            options["episode_id"] = torch.randint(1000000000, (1,)).expand(self.num_envs).to(self.env.device)  # [B]

        # 若提供评估种子，则在每个 env 的基础种子 self.seeds 上叠加该偏移
        if seed is not None:
            assert isinstance(seed, int), "seed must be an integer"
            reset_seeds = [base + seed for base in self.seeds]
            obs, info = self.env.reset(seed=reset_seeds, options=options)
        else:
            obs, info = self.env.reset(options=options)

        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = self.env.unwrapped.get_language_instruction()

        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(obs_image.device)  # [B, 1]

        for dq in self.endogenous_reward:
            dq.clear()

        return obs_image, instruction, info

    def step(self, raw_action):
        # Process actions with selective inference for incomplete environments only
        action = self._process_action_selective(raw_action)

        obs, _reward, _terminated, truncated, info = self.env.step(action)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        # truncated代表的含义是是否达到了最大步数，达到了为true，后面才进行统计
        truncated = truncated.reshape(-1, 1)  # [B, ] --> [B, 1]

        # Update success status
        self.success = self.success | (info["success"] & info["is_src_obj_grasped"]).cpu().numpy()

        if self.update_valid_envs:
            done = self.success.reshape(-1, 1)
        else:
            done = truncated

        # calculate reward - only for valid environments
        reward = self.get_reward(info)

        # process episode info
        # 所以这个地方是.all()才是更合适的写法 - 不过效果应该是一样的 这里truncated是统一的 都是step达到上限
        if truncated.any():
            done = truncated  # 和原版保持一致，rollout结束的地方都标记完成
            info["episode"] = {}
            # TODO 这里的统计可能有问题 dummy action填充后，success的环境最后这里可能会受影响吧?
            for k in ["is_src_obj_grasped", "consecutive_grasp"]:
                v = [info[k][idx].item() for idx in range(self.num_envs)]
                info["episode"][k] = v
            info["episode"]["success"] = self.success.astype(np.float32).tolist()

        return obs_image, reward, done, info

    def _process_action_selective(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """
        Process actions with selective inference - only process actions for incomplete environments.
        Use dummy actions for completed environments.

        Args:
            raw_actions: Raw action tokens from policy, only for incomplete environments

        Returns:
            Processed actions for all environments (including dummy actions for completed ones)
        """

        # Only process actions for valid (incomplete) environments
        if len(self.valid_envs_id) == self.num_envs:
            step_actions = self._process_action(raw_actions)
        else:
            # Initialize all actions with dummy actions
            dummy_action = get_maniskill_dummy_action()
            dummy_tensor = torch.tensor(dummy_action, dtype=torch.float64)  # same as predicted_step_action dtype
            step_actions = dummy_tensor.repeat(self.num_envs, 1)  # 根据环境数量生成dummy动作张量

            if len(self.valid_envs_id) > 0:
                predicted_step_action = self._process_action(raw_actions)

                # Update actions for incomplete environments
                for idx, action in enumerate(predicted_step_action):
                    env_id = self.valid_envs_id[idx]
                    assert not self.success[env_id]
                    step_actions[env_id] = action

        return step_actions

    def update_valid_envs_id(self):
        """Update the list of valid environment IDs."""
        if not self.update_valid_envs:
            return
        self.last_valid_envs_id = self.valid_envs_id
        self.valid_envs_id = [env_id for env_id, d in enumerate(self.success) if not d]

    def update_endogenous_reward(self, env_reward, endo_reward, valid_envs_id=None):
        """
        Update reward by combining environment reward with endogenous reward.

        Args:
            env_reward: Environment reward tensor [num_envs, 1]
            endo_reward: Endogenous reward tensor [num_valid_envs, 1] or None if use_endoRM=False

        Returns:
            Combined reward tensor [num_envs, 1]
        """
        # import pdb;pdb.set_trace()
        if not self.args.use_endoRM:
            # When endogenous reward is disabled, return only environment reward
            return env_reward
        if valid_envs_id is None:
            valid_envs_id = self.valid_envs_id
        if endo_reward is None:
            assert len(valid_envs_id) == 0
            return env_reward
        # When endogenous reward is enabled, combine both rewards
        # endo_reward only contains rewards for valid environments
        reward_tensor = torch.zeros(self.num_envs, 1).to(endo_reward.device)
        for i, env_id in enumerate(valid_envs_id):
            if self.use_avg_endo:
                # 将当前帧的endo_reward加入FIFO
                self.endogenous_reward[env_id].append(endo_reward[i].item())
                # 计算FIFO内的平均值
                endo_reward_i = sum(self.endogenous_reward[env_id]) / len(self.endogenous_reward[env_id])
            else:
                endo_reward_i = endo_reward[i].item()
            reward_tensor[env_id] += endo_reward_i

        reward_tensor += env_reward

        return reward_tensor
