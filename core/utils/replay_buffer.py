import torch
import numpy as np

class SeparatedReplayBuffer(object):
    def __init__(self, all_args, obs_dim, act_dim):
        self.ep_len = all_args.episode_len
        self.num_env = all_args.num_envs
        self.gamma = all_args.buffer_gamma
        self.gae_lambda = all_args.buffer_lambda
        self.buffer_minibatch = all_args.buffer_minibatch
        self.alg_grpo_fix = all_args.alg_grpo_fix
        self.update_valid_envs = all_args.update_valid_envs

        self.obs = np.zeros((self.ep_len + 1, self.num_env, *obs_dim), dtype=np.uint8)
        self.instruction = [""] * self.num_env
        self.value_preds = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.returns = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.actions = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.int32)
        self.action_log_probs = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.masks = np.ones((self.ep_len + 1, self.num_env, 1), dtype=np.float32)

        self.advantages = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)

        self.step = 0

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        # 这里为什么要这样处理？好像意思是buffer填满之后从头开始覆盖？ TODO 去掉 % 试一试
        self.step = (self.step + 1) % self.ep_len

    def warmup(self, obs, instruction):
        self.obs[0] = obs
        self.instruction = instruction
        self.masks[0] = 1.0

        self.step = 0

    def endup(self, next_value):
        self.value_preds[-1] = next_value

    def compute_returns_ppo(self):
        # 统一使用原版算法逻辑，确保兼容性
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            vt1 = self.value_preds[step + 1]
            vt = self.value_preds[step]
            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + vt

        # calc adv
        advantages = self.returns - self.value_preds[:-1]

        if self.update_valid_envs:
            # 使用masks来识别有效步骤（mask=1表示有效，mask=0表示环境已结束）
            valid_mask = (self.masks[:-1] == 1).squeeze(-1)  # 去掉最后一维，形状变为 (ep_len, num_env)

            if valid_mask.sum() > 0:  # 确保有有效数据
                valid_advantages = advantages[valid_mask]

                # 分布式训练时进行全局优势函数归一化
                import torch.distributed as dist
                if dist.is_initialized():
                    # 获取当前设备（CUDA设备用于分布式通信）
                    current_device = torch.cuda.current_device()
                    device = torch.device(f"cuda:{current_device}")

                    # 计算全局统计量（只使用有效数据）- 确保tensor在CUDA设备上
                    valid_advantages_tensor = torch.tensor(valid_advantages, dtype=torch.float32, device=device)

                    # 全局求和和计数 - 确保tensor在CUDA设备上
                    global_sum = valid_advantages_tensor.sum()
                    global_count = torch.tensor(valid_advantages_tensor.numel(), dtype=torch.float32, device=device)

                    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

                    global_mean = global_sum / global_count

                    # 计算全局方差
                    global_var_sum = ((valid_advantages_tensor - global_mean) ** 2).sum()
                    dist.all_reduce(global_var_sum, op=dist.ReduceOp.SUM)
                    global_std = torch.sqrt(global_var_sum / global_count)

                    # 使用全局统计量归一化
                    mean_advantages = global_mean.item()
                    std_advantages = global_std.item()
                else:
                    # 单卡训练时使用本地归一化
                    mean_advantages = valid_advantages.mean()
                    std_advantages = valid_advantages.std()

                # 对所有advantages进行标准化，但只使用有效部分的统计量
                self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
                # 将无效部分的advantages设为0
                self.advantages[~valid_mask] = 0
            else:
                # 如果没有有效数据，将所有advantages设为0
                self.advantages = np.zeros_like(advantages)
                import warnings
                warnings.warn("No valid steps found in replay buffer! All advantages set to zero. "
                             "This may indicate that all environments terminated at the first step.",
                             UserWarning)
        else:
            # 原版advantage计算（不过滤无效数据）
            # 分布式训练时进行全局优势函数归一化
            import torch.distributed as dist
            if dist.is_initialized():
                # 获取当前设备（CUDA设备用于分布式通信）
                current_device = torch.cuda.current_device()
                device = torch.device(f"cuda:{current_device}")

                # 计算全局统计量 - 确保tensor在CUDA设备上
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

                # 全局求和和计数 - 确保tensor在CUDA设备上
                global_sum = advantages_tensor.sum()
                global_count = torch.tensor(advantages_tensor.numel(), dtype=torch.float32, device=device)

                dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

                global_mean = global_sum / global_count

                # 计算全局方差
                global_var_sum = ((advantages_tensor - global_mean) ** 2).sum()
                dist.all_reduce(global_var_sum, op=dist.ReduceOp.SUM)
                global_std = torch.sqrt(global_var_sum / global_count)

                # 使用全局统计量归一化，最后转回numpy
                self.advantages = ((advantages_tensor - global_mean) / (global_std + 1e-5)).cpu().numpy()
            else:
                # 单卡训练时使用本地归一化
                mean_advantages = advantages.mean()
                std_advantages = advantages.std()
                self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    def compute_returns_grpo(self):
        # TODO valid filter
        if self.alg_grpo_fix:
            rewards_valid = self.rewards[self.rewards != 0]
            rewards_norm = self.rewards.copy()
            rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
            rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
        else:
            rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)

        returns = 0
        for step in reversed(range(self.rewards.shape[0])):
            returns = rewards_norm[step] + self.masks[step + 1] * returns
            self.returns[step] = returns

        # calc adv
        self.advantages = self.returns.copy()

    def get_minibatch_count(self):
        if self.update_valid_envs:
            # 只计算有效数据的minibatch数量
            valid_mask = (self.masks[:-1] == 1).reshape(-1)  # 形状: (episode_length * n_rollout_threads,)
            valid_batch_size = valid_mask.sum()

            if valid_batch_size == 0:
                return 1  # 避免除零错误

            if self.buffer_minibatch < 0:
                num_mini_batch = 1
            else:
                # 计算需要的minibatch数量（向上取整）
                num_mini_batch = (valid_batch_size + self.buffer_minibatch - 1) // self.buffer_minibatch

            return num_mini_batch
        else:
            # 原版逻辑
            episode_length, n_rollout_threads = self.rewards.shape[:2]
            batch_size = episode_length * n_rollout_threads

            if self.buffer_minibatch < 0:
                num_mini_batch = 1
            else:
                assert batch_size % self.buffer_minibatch == 0
                num_mini_batch = batch_size // self.buffer_minibatch

            return num_mini_batch

    def feed_forward_generator(self, target_mini_batches: int | None = None):
        """
        生成前馈训练 batch。
        - 当 update_valid_envs=True 时，按有效样本(valid masks)采样，并支持“重复采样补齐”。
        - 当提供 target_mini_batches 时，强制产出该数量的 mini-batches（通过重复采样补齐），用于多卡对齐 backward 次数。
        """
        # 统一使用原版逻辑作为基础，然后根据update_valid_envs决定是否过滤数据
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        # 预处理所有数据
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = self.advantages.reshape(-1, 1)

        if self.update_valid_envs:
            # 过滤出有效数据
            valid_mask = (self.masks[:-1] == 1).reshape(-1)  # 形状: (episode_length * n_rollout_threads,)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                import warnings
                warnings.warn("No valid data found in replay buffer for training!", UserWarning)
                return

            # 实际的有效batch size
            valid_batch_size = len(valid_indices)

            # 计算mini batch数量和大小
            if self.buffer_minibatch < 0:
                num_mini_batch = 1
                actual_minibatch_size = valid_batch_size
            else:
                actual_minibatch_size = self.buffer_minibatch
                # 计算自然的 mini-batch 数（向上取整）
                num_mini_batch = (valid_batch_size + actual_minibatch_size - 1) // actual_minibatch_size

            # 如果指定了目标 mini-batch 数，则提升到目标，并通过重复采样补齐索引
            if target_mini_batches is not None and target_mini_batches > num_mini_batch:
                num_mini_batch = target_mini_batches

            # 需要的总样本数
            total_needed = num_mini_batch * (actual_minibatch_size if self.buffer_minibatch > 0 else valid_batch_size)

            # 扩展 valid_indices 以补齐到 total_needed
            if total_needed > valid_batch_size:
                shortage = total_needed - valid_batch_size
                supplementary_indices = np.random.choice(valid_indices, size=shortage, replace=True)
                valid_indices = np.concatenate([valid_indices, supplementary_indices])
                valid_batch_size = total_needed

            # 随机打乱有效索引
            np.random.shuffle(valid_indices)

            if self.buffer_minibatch < 0:
                # 单批次：实际大小为有效数据量或 total_needed
                sampler = [valid_indices]
            else:
                sampler = [valid_indices[i * actual_minibatch_size:(i + 1) * actual_minibatch_size]
                           for i in range(num_mini_batch)]
        else:
            # 原版逻辑：使用所有数据
            if self.buffer_minibatch < 0:
                num_mini_batch = 1
                actual_minibatch_size = batch_size
            else:
                actual_minibatch_size = self.buffer_minibatch
                # 允许不整除：按随机打乱后补齐到整 mini-batch；并支持 target_mini_batches
                import math
                num_mini_batch = math.ceil(batch_size / actual_minibatch_size)

            import torch
            rand = torch.randperm(batch_size).numpy()

            # 若需要补齐到目标 mini-batch 数，或补满最后一个 mini-batch，执行重复采样填充
            total_needed = num_mini_batch * actual_minibatch_size
            if target_mini_batches is not None and target_mini_batches > num_mini_batch:
                num_mini_batch = target_mini_batches
                total_needed = num_mini_batch * actual_minibatch_size

            if total_needed > batch_size:
                shortage = total_needed - batch_size
                supplementary_indices = np.random.choice(rand, size=shortage, replace=True)
                rand = np.concatenate([rand, supplementary_indices])

            sampler = [rand[i * actual_minibatch_size:(i + 1) * actual_minibatch_size]
                       for i in range(num_mini_batch)]

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]

            # instruct
            instruct_indices = indices % n_rollout_threads
            instruct_batch = [self.instruction[i] for i in instruct_indices]

            yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)
