import os
import sys
import pprint
import random
import gc
import signal
import pdb
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated, Optional
import concurrent.futures

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import torch
import torch.distributed as dist
import numpy as np
import tyro
import wandb
from dataclasses import asdict, replace
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from core.utils.args import CLIArgs
from core.utils.utils import to_numpy, to_tensor, to_list, to_mean, to_tensor_device, parse_eval_seeds_arg
from core.utils.replay_buffer import SeparatedReplayBuffer
from core.utils.debug_rollout_vis import save_debug_rollout_videos, save_buffer_dump

import debugpy

"""
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
"""

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
# 在 encode 大量文本时会自动开 CPU 多线程，已经在使用多进程时易出错和爆内存，减少线程数量，提高资源可控性
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 全局 rank
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # 本地 rank
        local_rank = int(os.environ['LOCAL_RANK'])
        # Initialize process group
        from datetime import timedelta
        # 定义超时时间为 30 minutes
        timeout_duration = timedelta(minutes=30)
        dist.init_process_group(backend='nccl', timeout=timeout_duration, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def is_main_process():
    """if current process is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


class Runner:
    def __init__(self, all_args: CLIArgs):
        self.args = all_args
        self.benchmark = self.args.benchmark

        # 分布式初始化
        dist_info = init_distributed()
        if len(dist_info) == 3:
            self.rank, self.world_size, self.local_rank = dist_info
        else:
            self.rank, self.world_size = dist_info
            self.local_rank = 0
        if self.args.use_endoRM:
            # 解析reward_gpu_rank字符串为列表
            reward_ranks = [int(x) for x in self.args.reward_gpu_rank.split(',')]
            self.reward_local_rank = reward_ranks[self.local_rank]
        self.is_main = is_main_process()

        # set seed (不同rank不同seed)
        np.random.seed(self.args.seed + self.rank)
        random.seed(self.args.seed + self.rank)
        torch.manual_seed(self.args.seed + self.rank)

        # 初始化wandb和保存目录
        if self.is_main:
            # 主进程初始化wandb
            self.args.resume = self.args.resume and self.args.vla_load_path != ""
            wandb_init = False
            if self.args.resume:
                wandb_id = self.args.vla_load_path.split("/")[-3]
                if wandb_id.startswith("run-"):
                    wandb_id = wandb_id.split("-")[-1]
                    wandb.init(
                        config=all_args.__dict__,
                        project="RLVLA",
                        name=self.args.name,
                        mode="online" if self.args.wandb else "offline",
                        resume="must",
                        id=wandb_id
                    )
                    print(f"----------Resuming wandb run id: {wandb_id}")
                    wandb_init = True
            if not wandb_init:
                wandb.init(
                    config=all_args.__dict__,
                    project="RLVLA",
                    name=self.args.name,
                    mode="online" if self.args.wandb else "offline"
                )
            wandb.run.tags = wandb.run.tags + (wandb.run.id,)
            self.save_dir = Path(wandb.run.dir)
            self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
            self.glob_dir.mkdir(parents=True, exist_ok=True)
            yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

            # 构造要广播的信息（仅包含可序列化、各 rank 都能用到的字段）
            wandb_info = {
                "run_id": str(wandb.run.id),
                "run_name": str(wandb.run.name),
                "run_dir": str(wandb.run.dir),
                "glob_dir": str(self.glob_dir),
            }
        else:
            # 非主进程等待接收wandb信息
            wandb_info = None

        # 分布式广播：将 rank0 的 wandb_info 同步给所有 rank
        if dist.is_initialized() and dist.get_world_size() > 1 and self.is_main:
            obj_list = [wandb_info]
            dist.broadcast_object_list(obj_list, src=0)  # 自带同步语义，无需另行 barrier
            wandb_info = obj_list[0]
        else:
            # 单进程情况直接使用主进程信息
            assert wandb_info is not None, "wandb_info should not be None in single-process mode."

        # 每个进程本地设置（不要在非主进程访问 wandb.run）
        self.wandb_run_id = wandb_info["run_id"]
        self.wandb_run_name = wandb_info["run_name"]
        self.wandb_run_dir = wandb_info["run_dir"]
        self.glob_dir = Path(wandb_info["glob_dir"])

        # policy
        # NOTE 放在一开始会报错，accelerate.PartialState 在模块导入时自动初始化分布式进程组，会导致重复init
        from core.policy.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        self.ddp_device = torch.device(f"cuda:{self.local_rank}")
        self.policy = OpenVLAPolicy(all_args, self.ddp_device)

        if self.args.use_endoRM and not self.args.self_reward:
            self.reward_device = torch.device(f"cuda:{self.reward_local_rank}")
            # 新增：reward model加载路径优先使用reward_model_path，否则用vla_path
            reward_model_path = self.args.reward_model_path if self.args.reward_model_path else self.args.vla_path
            reward_model_lora_path = self.args.reward_vla_load_path if self.args.reward_vla_load_path else self.args.vla_load_path
            # 传递reward_model_path给OpenVLAPolicy（如有需要可在OpenVLAPolicy中处理）
            reward_args = replace(all_args, vla_path=reward_model_path)
            reward_args = replace(reward_args, vla_load_path=reward_model_lora_path)
            self.reward_model = OpenVLAPolicy(reward_args, self.reward_device)
            # 确保 reward_model 仅用于推理：设置 eval 并冻结所有参数，避免被任何优化器或反向传播更新
            self.reward_model.vla.eval()
            for p in self.reward_model.vla.parameters():
                p.requires_grad = False
            # 释放并禁用 reward_model 上不必要的优化器状态，减少内存占用
            if hasattr(self.reward_model, "vh_optimizer"):
                self.reward_model.vh_optimizer = None
            if hasattr(self.reward_model, "vla_optimizer"):
                self.reward_model.vla_optimizer = None

        self.steps = self.policy.start_epoch
        self.start_epoch = self.policy.start_epoch
        # best model tracking
        self.best_epoch = self.policy.best_epoch
        self.best_metric_value = self.policy.best_metric_value

        # 分布式模型包装
        if dist.is_initialized():
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.policy.vla = DDP(
                self.policy.vla,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )

        # 可选：使用方案一（全局汇总再分配版本）
        self.use_global_ppo = getattr(self.args, 'use_global_ppo', False)
        if self.use_global_ppo:
            from core.policy.openvla.openvla_train_global import OpenVLAPPOGlobal
            self.alg = OpenVLAPPOGlobal(all_args, self.policy)
        else:
            self.alg = OpenVLAPPO(all_args, self.policy)

        # 环境分配，每个进程分一部分envs
        if dist.is_initialized():
            all_num_envs = self.args.num_envs * self.world_size
            if self.is_main:
                print(f"[DDP] world_size={self.world_size}, num_envs per proc={self.args.num_envs}, all num={all_num_envs}")
        # 获取unnorm_state，DDP包装后需要通过.module访问
        if dist.is_initialized():
            unnorm_state = self.policy.vla.module.get_action_stats(self.args.vla_unnorm_key)
        else:
            unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)

        # 获取环境类型
        env_backend = getattr(self.args, 'benchmark', 'maniskill').lower()
        # 动态导入
        if env_backend == 'maniskill':
            from core.benchmark.maniskill.maniskill_wrapper import ManiSkillWrapper
            self.env = ManiSkillWrapper(self.args, unnorm_state, extra_seed=self.local_rank, device_id=self.local_rank)
        elif env_backend == 'libero':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
            from core.benchmark.libero.libero_wrapper import LiberoWrapper
            self.env = LiberoWrapper(self.args, unnorm_state, extra_seed=self.local_rank, device_id=self.local_rank)
        else:
            raise ValueError(f"Unknown env_backend: {env_backend}. Supported: 'simler', 'libero'.")

        self.args.episode_len = self.env.episode_len

        # buffer
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(self.args.img_height, self.args.img_width, 3),
            act_dim=7,
        )
        # 新增：独立保存 logits 的容器（按需懒初始化），形状 [T, N, L, V]
        self.logits_store = None
        minibatch_count = self.buffer.get_minibatch_count()
        if self.is_main:
            print(f"Max buffer minibatch count: {minibatch_count}")

        # 异步执行相关初始化
        # 检查是否启用异步执行（可以通过环境变量或参数控制）
        # 当启用 self_reward 时，强制禁用异步模式，使用同步逻辑
        self.async_enabled = (
            self.args.use_endoRM and
            self.args.async_enabled and
            not self.args.self_reward and  # self_reward 时禁用异步
            os.environ.get('DISABLE_ASYNC_ENDO_REWARD', '').lower() != 'true'
        )

        if self.async_enabled:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="endo_reward"
            )
            self.pending_endo_reward_future: Optional[concurrent.futures.Future] = None
            self.pending_data: Optional[dict] = None  # 存储待组合的数据

            if self.is_main:
                print(f"[INFO] Async endo_reward execution enabled (delayed combination mode)")
        else:
            self.executor = None
            self.pending_endo_reward_future = None
            self.pending_data = None

            if self.is_main and self.args.use_endoRM:
                print(f"[INFO] Async endo_reward execution disabled, using synchronous mode")


    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []
        logits_tensors = []

        # TODO adjust parameter to avoid too many batches
        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob, logits_tensor = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
            logits_tensors.append(logits_tensor)

        values = torch.cat(values, dim=0).to(device=self.ddp_device)
        actions = torch.cat(actions, dim=0).to(device=self.ddp_device)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.ddp_device)
        logits_tensors = torch.cat(logits_tensors, dim=0).to(device=self.ddp_device)

        return values, actions, logprobs, logits_tensors

    # 使用 inference_mode 比 no_grad 更省开销，禁用 Autograd 并避免版本计数开销
    @torch.inference_mode()
    def _get_endo_reward(self, obs, action):
        if not self.args.use_endoRM:
            return None

        self.reward_model.prep_rollout()

        total_batch = obs["image"].shape[0]
        obs["image"] = obs["image"].to(self.reward_device)
        action = action.to(self.reward_device)

        endo_rewards = []
        for i in range(0, total_batch, self.args.buffer_inferbatch):
            batch_slice = slice(i, i + self.args.buffer_inferbatch)
            obs_batch = {k: v[batch_slice] for k, v in obs.items()}
            actions_batch = action[batch_slice]

            reward = self.reward_model.get_endo_reward(obs_batch, actions_batch)
            endo_rewards.append(reward/self.args.endo_reward_scale)

        endo_rewards = torch.cat(endo_rewards, dim=0).to(device=self.ddp_device)

        return endo_rewards

    @torch.inference_mode()
    def _get_self_reward(self, action, logits_tensors):
        """
        基于已有的 logits_tensors 和 actions 直接计算内生奖励
        不需要模型前向推理，不需要分batch处理
        """
        if not self.args.self_reward:
            return None

        action = action.to(self.ddp_device)
        logits_tensors = logits_tensors.to(self.ddp_device)

        # 参考 get_endo_reward 的计算逻辑
        coef = self.args.endo_reward_reg_coef

        # logits_tensors 形状是 [B, action_len, 256]（已经是动作词汇表的 logits）
        # 温度缩放 + softmax
        logits_tensor = logits_tensors / coef
        logprobs_tensor = torch.log_softmax(logits_tensor, dim=-1)  # [B, action_len, 256]

        # 计算实际动作的 log 概率
        # action 中的动作 token 需要转换为相对于动作词汇表 [0, 255] 的索引
        # 动作 token 范围是 [32000-256, 32000)，需要减去 (32000-256) 得到 [0, 255]
        action_indices = (action - (32000 - 256)).unsqueeze(-1)  # [B, action_len, 1]
        action_indices = action_indices.to(logprobs_tensor.device)

        # 按照索引取出每个动作位置上实际生成的动作 token 的 log 概率
        logprobs = torch.gather(logprobs_tensor, 2, action_indices).squeeze(-1)  # [B, action_len]
        logprobs_sum = coef * logprobs.sum(dim=1, keepdim=True)  # [B, 1]

        if not self.args.with_VQ:
            endo_rewards = logprobs_sum
        else:
            # 计算 V_Q_1（如果启用）
            # 取出第一个动作位置对应的 logits
            logits_a0 = logits_tensors[:, 0, :]  # [B, 256]
            logits_a0_scaled = logits_a0 / coef
            V_Q_1 = coef * torch.logsumexp(logits_a0_scaled, dim=-1, keepdim=True)  # [B, 1]
            endo_rewards = logprobs_sum + V_Q_1

        # 应用缩放系数
        endo_rewards = endo_rewards / self.args.endo_reward_scale

        return endo_rewards

    @torch.no_grad()
    def _get_endo_reward_bk(self, obs, action):
        if not self.args.use_endoRM:
            return None

        self.reward_model.prep_rollout()

        total_batch = obs["image"].shape[0]
        obs["image"] = obs["image"].to(self.reward_device)
        action = action.to(self.reward_device)

        endo_rewards = []
        # import time
        # start_time = time.time()

        # TODO adjust parameter to avoid too many batches
        for i in range(0, total_batch, self.args.buffer_inferbatch):
            batch_slice = slice(i, i + self.args.buffer_inferbatch)
            obs_batch = {k: v[batch_slice] for k, v in obs.items()}
            actions_batch = action[batch_slice]

            # logprob, entropy, values = self.reward_model.evaluate_actions(obs_batch, actions_batch)

            with torch.no_grad():
                endo_reward = self.reward_model.get_endoreward_bk(obs_batch, actions_batch)
            endo_rewards.append(endo_reward)

        endo_rewards = torch.cat(endo_rewards, dim=0).to(device=self.ddp_device)

        # wait_time = time.time() - start_time
        # print(f"_get_endo_reward 耗时: {wait_time:.3f} 秒")
        return endo_rewards

    def _wait_for_pending_endo_reward(self) -> Optional[torch.Tensor]:
        """
        等待并获取异步 endo_reward 计算结果（阻塞）
        如果异步任务失败或超时，会直接抛出异常
        """
        if self.pending_endo_reward_future is None:
            return None

        # 等待异步任务完成，如果超时或失败会直接抛出异常
        result = self.pending_endo_reward_future.result(timeout=100.0)
        self.pending_endo_reward_future = None

        return result

    def _start_async_endo_reward(self, obs, action):
        """
        启动异步 endo_reward 计算
        如果启动失败会直接抛出异常
        """
        if not self.async_enabled:
            return

        # 深拷贝数据避免竞态条件
        obs_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() if isinstance(v, list) else v
                   for k, v in obs.items()}
        action_copy = action.clone()

        # 提交异步任务，如果失败会直接抛出异常
        self.pending_endo_reward_future = self.executor.submit(
            self._get_endo_reward, obs_copy, action_copy
        )



    def collect(self, step=None):
        self.policy.prep_rollout()

        if step is None:
            if self.async_enabled and self.pending_data is not None:
                # 异步的逻辑下，buffer insert要滞后一个step，不能使用
                obs_image = self.pending_data["obs_img"]
            else:
                obs_image = self.buffer.obs[self.buffer.step]  # 不能是-1, 因为初始化是numpy 不是list
        else:
            obs_image = self.buffer.obs[step]

        # filter unvalid obs
        if len(self.env.valid_envs_id) == 0:
            # All environments have completed their tasks successfully
            # Return dummy values with appropriate shapes to maintain training flow
            if step is None:
                print(f"All environments completed successfully in rank {self.rank}, returning dummy values")

            # Create zero tensors with correct shapes efficiently (avoid unnecessary forward pass)
            # Use the expected shapes based on environment configuration
            batch_size = self.env.num_envs
            action_dim = 7  # From buffer initialization

            dummy_value = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.ddp_device)
            dummy_action = torch.zeros((batch_size, action_dim), dtype=torch.int32, device=self.ddp_device)
            dummy_logprob = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=self.ddp_device)

            endo_reward = None
            if self.async_enabled:
                # previous_endo_reward
                endo_reward = self._wait_for_pending_endo_reward()

            return dummy_value, dummy_action, dummy_logprob, endo_reward, None

        elif len(self.env.valid_envs_id) == self.env.num_envs:
            valid_obs_image = obs_image
            valid_task_description = self.buffer.instruction
        else:
            valid_obs_image = []
            valid_task_description = []
            for valid_env_id in self.env.valid_envs_id:
                valid_obs_image.append(obs_image[valid_env_id])
                valid_task_description.append(self.buffer.instruction[valid_env_id])

        # Creating a tensor from a list of numpy.ndarrays is extremely slow
        # obs_image = torch.tensor(valid_obs_image).to(self.device)
        # 支持 valid_obs_image 为 tensor、ndarry, list[tensor]、list[array]
        # 使用通用工具函数将各种输入类型转换为目标 device 上的张量
        obs_image = to_tensor_device(valid_obs_image, self.ddp_device)
        obs = dict(image=obs_image, task_description=valid_task_description)

        # 异步执行逻辑（延迟组合策略）
        if self.async_enabled and step is None:
            # 1. 执行当前轮的 action 计算（与上一轮的 endo_reward 计算并行）
            # import time
            # start_time = time.time()
            value, action, logprob, logits_tensors = self._get_action(obs)
            # wait_time = time.time() - start_time
            # print(f"get action 耗时: {wait_time:.3f} 秒")

            # 2. 获取上一轮的 endo_reward（阻塞）
            # start_time = time.time()
            previous_endo_reward = self._wait_for_pending_endo_reward()
            # wait_time = time.time() - start_time
            # print(f"[异步 endo_reward] 等待耗时: {wait_time:.3f} 秒")

            # 3. 启动当前轮的 endo_reward 异步计算
            # start_time = time.time()
            self._start_async_endo_reward(obs, action)
            # wait_time = time.time() - start_time
            # print(f"[异步 endo_reward] 启动耗时: {wait_time:.3f} 秒")

            # 4. 返回上一轮的 endo_reward，用于延迟组合
            endo_reward = previous_endo_reward

        else:
            # 原始同步执行逻辑
            value, action, logprob, logits_tensors = self._get_action(obs)
            if step is None:
                if self.args.self_reward:
                    # 使用训练模型自身和 logits_tensors 计算内生奖励
                    endo_reward = self._get_self_reward(action, logits_tensors)
                else:
                    # 使用独立的 reward_model 计算内生奖励
                    endo_reward = self._get_endo_reward(obs, action)
            else:
                endo_reward = None

        # padding 已移动到 libero_wrapper 的 step 函数中

        return value, action, logprob, endo_reward, logits_tensors  # data of valid envs

    def warmup(self, obs_img, instruction):
        obs_img = to_numpy(obs_img)
        self.buffer.warmup(obs_img, instruction)

    def insert(self, data, valid_envs_id=None, logits_tensors=None):
        # 记录当前 step（buffer.insert 内部会自增 step）
        current_step = self.buffer.step
        if valid_envs_id is None:
            valid_envs_id = self.env.valid_envs_id
        obs_img, actions, logprob, value_preds, rewards, done = data

        obs_img = to_numpy(obs_img)
        actions = to_numpy(actions, np.int32)
        logprob = to_numpy(logprob, np.float32)
        value_preds = to_numpy(value_preds, np.float32)
        rewards = to_numpy(rewards, np.float32)
        done = to_numpy(done, np.float32)
        masks = 1.0 - done  # 1 means valid (env not finished)
        # convert logits to numpy (None stays None)
        logits_np = to_numpy(logits_tensors, np.float32) if logits_tensors is not None else None


        # padding 逻辑：obs_img, rewards, done 已在 libero_wrapper 的 step 函数中处理
        # 但 actions, logprob, value_preds 来自 collect()，仍需要在这里进行 padding；logits 同步对齐
        if len(valid_envs_id) < self.env.num_envs:
            # Handle edge case when all environments are completed (valid_envs_id is empty)
            if len(valid_envs_id) == 0:
                # All data should be zeros, use fixed shapes
                full_actions = np.zeros((self.args.num_envs, 7), dtype=np.int32)  # action_dim=7
                full_logprob = np.zeros((self.args.num_envs, 7), dtype=np.float32)
                full_value_preds = np.zeros((self.args.num_envs, 1), dtype=np.float32)
                full_logits = None if logits_np is None else np.zeros((self.args.num_envs, *logits_np.shape[1:]), dtype=np.float32)
            else:
                # Normal case: some environments are still active
                full_actions = np.zeros((self.args.num_envs, *actions.shape[1:]), dtype=actions.dtype)
                full_logprob = np.zeros((self.args.num_envs, *logprob.shape[1:]), dtype=logprob.dtype)
                full_value_preds = np.zeros((self.args.num_envs, *value_preds.shape[1:]), dtype=value_preds.dtype)

                # 将有效环境的数据填入对应位置
                for i, valid_env_id in enumerate(valid_envs_id):
                    full_actions[valid_env_id] = actions[i]
                    full_logprob[valid_env_id] = logprob[i]
                    full_value_preds[valid_env_id] = value_preds[i]

                # 同步构造 logits 的 padding
                if logits_np is not None:
                    full_logits = np.zeros((self.args.num_envs, *logits_np.shape[1:]), dtype=np.float32)
                    for i, valid_env_id in enumerate(valid_envs_id):
                        full_logits[valid_env_id] = logits_np[i]
                else:
                    full_logits = None

            # 新增：将本 step 的 logits 保存到独立容器
            if full_logits is not None:
                if self.logits_store is None:
                    # 懒初始化：使用当前 logits 的动作长度与词表大小
                    action_len, vocab_size = full_logits.shape[1], full_logits.shape[2]
                    self.logits_store = np.zeros((self.args.episode_len, self.args.num_envs, action_len, vocab_size), dtype=np.float32)
                self.logits_store[current_step] = full_logits
            self.buffer.insert(obs_img, full_actions, full_logprob, full_value_preds, rewards, masks)
        else:
            # 新增：将本 step 的 logits 保存到独立容器
            if logits_np is not None:
                if self.logits_store is None:
                    action_len, vocab_size = logits_np.shape[1], logits_np.shape[2]
                    self.logits_store = np.zeros((self.args.episode_len, self.args.num_envs, action_len, vocab_size), dtype=np.float32)
                self.logits_store[current_step] = logits_np
            self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks)

        self.env.update_valid_envs_id()

    def compute_endup(self):
        with torch.no_grad():
            next_value, _, _, _, _ = self.collect(step=-1)  # 这里好像也不需要用step=-1 就正常的collect就够了
        next_value = next_value.to(torch.float32).cpu().numpy()

        # padding - 恢复next_value到完整的num_envs大小
        full_next_value = np.zeros((self.args.num_envs, *next_value.shape[1:]), dtype=next_value.dtype)

        # 处理有效环境的next_value
        if len(self.env.valid_envs_id) > 0:
            # 将有效环境的next_value填入对应位置
            for i, valid_env_id in enumerate(self.env.valid_envs_id):
                full_next_value[valid_env_id] = next_value[i]

        # 为无效环境id填充默认值（0.0表示无价值）
        invalid_env_ids = [i for i in range(self.args.num_envs) if i not in self.env.valid_envs_id]
        for invalid_env_id in invalid_env_ids:
            full_next_value[invalid_env_id] = 0.0

        self.buffer.endup(full_next_value)

    def train(self, episode_progress=None):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer, progress=episode_progress)
        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")

        info = {f"train/{k}": v for k, v in train_info.items()}
        info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
        info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)

        return info

    def distributed_mean_dict(self, info: dict):
        if not dist.is_initialized():
            return info
        try:
            # print(f"[rank {self.rank}] before all_reduce")
            keys = sorted(info.keys())
            values = torch.tensor([float(info[k]) for k in keys], device=self.ddp_device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= dist.get_world_size()
            # print(f"[rank {self.rank}] after all_reduce")
            return dict(zip(keys, values.cpu().numpy()))
        except Exception as e:
            print(f"[rank {self.rank}] all_reduce failed: {e}")
            import sys
            sys.exit(1)

    @torch.no_grad()
    def eval(self, env_type: str, seeds: Optional[list[int]] = None) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])

        # 若未提供 seeds，则保持默认 reset 行为；若提供，则按长度循环评估
        if seeds is None:
            obs_img, instruction, info = self.env.reset(env_type=env_type)
            episode_len = self.env.episode_len
            obs_img = to_tensor(obs_img)

            for _ in range(episode_len):
                obs = dict(image=obs_img, task_description=instruction)
                value, action, logprob, logits_tensors = self._get_action(obs, deterministic=True)

                obs_img, reward, done, env_info = self.env.step(action)
                obs_img = to_tensor(obs_img)

                # info 聚合
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        tmp_val = to_list(v)
                        if isinstance(env_infos[f"{k}"], list):
                            env_infos[f"{k}"].extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                        else:
                            cur = to_list(env_infos[f"{k}"])
                            cur = cur if isinstance(cur, list) else [cur]
                            cur.extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                            env_infos[f"{k}"] = cur
        else:
            for seed in seeds:
                # 固定环境种子进行一次评估（ManiSkill / LIBERO wrapper 内部处理每个 env 的具体 seed 列表）
                obs_img, instruction, info = self.env.reset(env_type=env_type, seed=seed)
                episode_len = self.env.episode_len
                obs_img = to_tensor(obs_img)

                for _ in range(episode_len):
                    obs = dict(image=obs_img, task_description=instruction)
                    value, action, logprob, logits_tensors = self._get_action(obs, deterministic=True)

                    obs_img, reward, done, env_info = self.env.step(action)
                    obs_img = to_tensor(obs_img)

                    # info 聚合（跨多次 seeds 追加）
                    if "episode" in env_info.keys():
                        for k, v in env_info["episode"].items():
                            tmp_val = to_list(v)
                            if isinstance(env_infos[f"{k}"], list):
                                env_infos[f"{k}"].extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                            else:
                                cur = to_list(env_infos[f"{k}"])
                                cur = cur if isinstance(cur, list) else [cur]
                                cur.extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                                env_infos[f"{k}"] = cur

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()
        return env_stats

    @torch.no_grad()
    def render(self, epoch: int, env_type: str) -> dict:
        # 只在主进程执行渲染和保存
        if not self.is_main:
            return {}

        self.policy.prep_rollout()

        seeds_for_render = self.args.eval_seeds or list(range(5))
        if len(seeds_for_render) == 0:
            seeds_for_render = list(range(5))
        num_cycles = len(seeds_for_render)

        # 初始化汇总用的环境信息字典（所有循环都会累积到这里）
        total_env_infos = defaultdict(lambda: [])
        
        # 针对每个 eval seed 运行一次 render 循环
        for cycle_idx, seed in enumerate(seeds_for_render, start=1):
            print(f"Starting render cycle {cycle_idx}/{num_cycles} (seed={seed})")
            # 每次循环初始化当前周期的日志
            current_datas = [{
                "image": [],  # obs_t: [0, T-1]
                "instruction": "",
                "action": [],  # a_t: [0, T-1]
                "info": [],  # info after executing a_t: [1, T]
            } for idx in range(self.args.num_envs)]

            # 重置环境，获取初始观测
            obs_img, instruction, info = self.env.reset(env_type=env_type, seed=seed)
            episode_len = self.env.episode_len
            obs_img = to_tensor(obs_img)
            print(f"Seed {seed} instruction[:3]:", instruction[:3])

            # 记录每个环境的指令
            for idx in range(self.args.num_envs):
                current_datas[idx]["instruction"] = instruction[idx]

            # 运行当前周期的episode
            for _ in range(episode_len):
                obs = dict(image=obs_img, task_description=instruction)
                value, action, logprob, logits_tensors = self._get_action(obs, deterministic=True)

                obs_img_new, reward, done, env_info = self.env.step(action)
                obs_img_new = to_tensor(obs_img_new)

                # 打印当前步骤信息
                print({k: round(to_mean(v), 4) if isinstance(to_mean(v), (int, float, np.integer, np.floating)) else to_mean(v)
                    for k, v in env_info.items() if k != "episode"})

                # 累积环境信息到总字典中
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        tmp_val = to_list(v)
                        # 直接累加到总环境信息中，而不是当前周期的
                        if isinstance(total_env_infos[f"{k}"], list):
                            total_env_infos[f"{k}"].extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                        else:
                            cur = to_list(total_env_infos[f"{k}"])
                            cur = cur if isinstance(cur, list) else [cur]
                            cur.extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                            total_env_infos[f"{k}"] = cur

                # 记录当前周期的数据
                for i in range(self.args.num_envs):
                    post_action = self.env._process_action(action)
                    log_image = to_numpy(obs_img[i])
                    log_action = to_numpy(post_action[i]).tolist()
                    log_info = {k: to_list(v[i]) for k, v in env_info.items() if k != "episode"}
                    current_datas[i]["image"].append(log_image)
                    current_datas[i]["action"].append(log_action)
                    current_datas[i]["info"].append(log_info)

                # 更新观测
                obs_img = obs_img_new

            # 记录每个环境的最后一帧图像
            for i in range(self.args.num_envs):
                log_image = obs_img[i].cpu().numpy()
                current_datas[i]["image"].append(log_image)

            # 创建当前周期的保存目录（按 seed 命名）
            exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{env_type}_rank-{self.rank}_seed-{seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            # 保存当前周期的视频
            for i in range(self.args.num_envs):
                images = current_datas[i]["image"]
                infos = current_datas[i]["info"]
                assert len(images) == len(infos) + 1, f"Seed {seed} env {i} image-info length mismatch"

                if self.args.render_info:
                    for j in range(len(infos)):
                        images[j + 1] = visualization.put_info_on_image(
                            images[j + 1], infos[j],
                            extras=[f"Ins: {instruction[i]}", f"Seed: {seed}"]
                        )

                success = int(infos[-1]["success"])
                # 视频文件名包含周期编号和环境编号
                images_to_video(images, str(exp_dir), f"video_seed-{seed}_env-{i}-s_{success}",
                                fps=10, verbose=False)

            print(f"Completed render cycle {cycle_idx}/{num_cycles} (seed={seed})\n")

        # 计算所有循环的汇总统计信息
        env_stats = {k: np.mean(v) for k, v in total_env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(f"汇总 {num_cycles} 个种子的统计信息:")
        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # 保存汇总统计信息
        summary_dir = Path(self.glob_dir) / f"vis_{epoch}_{env_type}_rank-{self.rank}_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # 构建汇总统计数据
        max_entries = self.args.num_envs * num_cycles
        last_info = {
            idx: {
                k: total_env_infos[k][idx]
                for k in total_env_infos.keys()
                if isinstance(total_env_infos[k], list) and len(total_env_infos[k]) > idx
            }
            for idx in range(max_entries)
        }

        save_stats = {
            "env_name": self.args.env_id,
            "ep_len": self.args.episode_len,
            "epoch": epoch,
            "total_cycles": num_cycles,
            "total_envs": self.args.num_envs * num_cycles,
            "render_seeds": seeds_for_render,
            "stats": {k: v.item() for k, v in env_stats.items()},
            "last_info": last_info
        }

        yaml.dump(save_stats, open(summary_dir / "summary_stats.yaml", "w"))

        return env_stats_ret


    def cleanup_async_resources(self):
        """
        清理异步执行相关的资源
        如果等待异步任务失败会直接抛出异常
        """
        if hasattr(self, 'executor') and self.executor is not None:
            # 等待所有挂起的任务完成，如果超时或失败会直接抛出异常
            if self.pending_endo_reward_future is not None:
                self.pending_endo_reward_future.result(timeout=100.0)

            # 关闭线程池
            self.executor.shutdown(wait=True)
            if self.is_main:
                print("Async executor shutdown completed")

    def reset_async_state(self):
        """
        重置异步执行状态，通常在每个 episode 开始时调用
        如果等待异步任务失败会直接抛出异常
        """
        if self.async_enabled:
            # 等待任何挂起的异步任务完成，如果超时或失败会直接抛出异常
            if self.pending_endo_reward_future is not None:
                self.pending_endo_reward_future.result(timeout=100.0)
                self.pending_endo_reward_future = None

            # 清理待处理数据
            self.pending_data = None

    def run(self):
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        # 外层episode进度条（只在rank 0显示，position=0）
        for episode in tqdm(
                range(self.start_epoch, max_episodes),
                initial=self.start_epoch,
                total=max_episodes,
                desc="episode",
                position=0,
                disable=(self.rank != 0),
                leave=True):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()

            obs_img, instruction, info = self.env.reset(env_type="train", same_init=self.args.use_same_init)
            episode_len = self.env.episode_len

            # 重置异步状态，确保每个 episode 开始时状态干净
            self.reset_async_state()

            self.warmup(obs_img, instruction)
            # 重置 logits 独立存储（每个 episode 重新开始记录）
            self.logits_store = None
            # 全局 episode 进度；考虑断点续训，范围[0,1]
            denom = max(1, (max_episodes - self.start_epoch))
            episode_progress = (episode - self.start_epoch + 1) / denom
            episode_progress = min(max(episode_progress, 0.0), 1.0)

            # 内层rollout进度条（每个rank一行，只有rank 0 leave=True，其余leave=False）
            for step in tqdm(
                    range(episode_len),
                    desc=f"rollout [rank {self.rank}]",
                    position=self.rank + 1,
                    leave=(self.rank == 0)):
                # 同步时，返回的都是当前step；异步时，value, action, logprob为当前step，endo_reward为上一step
                value, action, logprob, endo_reward, logits_tensors = self.collect()

                obs_img, env_reward, done, env_info = self.env.step(action)

                # 延迟组合策略：处理数据的时序对应
                if self.async_enabled:
                    # 如果有待处理的数据，先处理上一轮的数据
                    if self.pending_data is not None:
                        # 组合上一轮的 env_reward 和当前获得的 endo_reward
                        prev_reward = self.env.update_endogenous_reward(
                            self.pending_data['env_reward'], endo_reward, self.env.last_valid_envs_id
                        )

                        # 插入上一轮的完整数据
                        prev_data = (
                            self.pending_data['obs_img'],
                            self.pending_data['action'],
                            self.pending_data['logprob'],
                            self.pending_data['value'],
                            prev_reward,
                            self.pending_data['done']
                        )
                        self.insert(prev_data, self.env.last_valid_envs_id, logits_tensors=self.pending_data.get('logits'))

                    # 存储当前轮的数据，等待下一轮的 endo_reward
                    self.pending_data = {
                        'obs_img': obs_img,
                        'action': action,
                        'logprob': logprob,
                        'value': value,
                        'env_reward': env_reward,
                        'done': done,
                        'logits': logits_tensors,
                    }

                else:
                    # 同步模式：直接组合并插入
                    reward = self.env.update_endogenous_reward(env_reward, endo_reward)
                    data = (obs_img, action, logprob, value, reward, done)
                    self.insert(data, logits_tensors=logits_tensors)

                # info
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        # 兼容list / numpy / tensor，强制将env_infos[k]维护为Python列表，避免numpy数组的就地相加导致广播错误
                        tmp_val = to_list(v)
                        if isinstance(env_infos[f"{k}"], list):
                            env_infos[f"{k}"].extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                        else:
                            cur = to_list(env_infos[f"{k}"])
                            cur = cur if isinstance(cur, list) else [cur]
                            cur.extend(tmp_val if isinstance(tmp_val, list) else [tmp_val])
                            env_infos[f"{k}"] = cur

            print(f"[rank {self.rank}] " + pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))

            # 异步模式：处理最后一轮的待处理数据
            if self.async_enabled:
                assert self.pending_data is not None
                # 等待最后一轮的 endo_reward 计算完成
                final_endo_reward = self._wait_for_pending_endo_reward()

                # 组合最后一轮的数据
                final_reward = self.env.update_endogenous_reward(
                    self.pending_data['env_reward'], final_endo_reward, self.env.last_valid_envs_id
                )

                # 插入最后一轮的数据
                final_data = (
                    self.pending_data['obs_img'],
                    self.pending_data['action'],
                    self.pending_data['logprob'],
                    self.pending_data['value'],
                    final_reward,
                    self.pending_data['done']
                )
                self.insert(final_data, self.env.last_valid_envs_id, logits_tensors=self.pending_data.get('logits'))

                # 清理待处理数据
                self.pending_data = None

            # train and process infos
            # 最后一步的 value（即 episode 结束时的 value），在采样时还没算出来，需要在 episode 结束后单独补上
            self.compute_endup()

            # 以下步骤似乎不是很必要
            del value, action, logprob, obs_img, done
            # CPU内存释放
            # gc.collect()
            # GPU内存释放
            torch.cuda.empty_cache()

            if dist.is_initialized():
                dist.barrier()  # 所有进程同步等待

            # save debug rollout videos before training
            save_debug_rollout_videos(self.args, self.buffer, self.glob_dir, episode, self.rank)

            # train
            infos = self.train(episode_progress)

            # 为了计算advantage，在train后保存buffer信息
            if self.is_main:
                # 从env_infos中提取成功标记（若存在），用于分类分析；否则在保存函数中根据mask推断完成情况
                success_array = np.array(env_infos.get('success'), dtype=bool) if 'success' in env_infos else None
                save_buffer_dump(self.args, self.buffer, self.glob_dir, episode, self.rank, success_array, logits_external=self.logits_store)

            # env_infos，后面会调用self.distributed_mean_dict同步到所有进程
            env_stats = {f"env/{k}": np.mean(v) for k, v in env_infos.items()}
            infos.update(env_stats)

            # steps
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs * self.args.alg_ppo_epoch

            if self.is_main:
                # 同步所有进程的infos，用于wandb记录
                if dist.is_initialized():
                    infos = self.distributed_mean_dict(infos)
                if 'train/train_steps' in infos:
                    self.steps += int(infos['train/train_steps'])  # 统计early_stop实际的step
                    steps = self.steps
                    del infos['train/train_steps']  # 删除steps，避免重复记录
                wandb.log(infos, step=steps)
                elapsed_time = time.time() - ep_time
                print(f"{self.args.name}: ep {episode:0>4d} | steps {steps} | e {elapsed_time:.2f}s")
                print(pprint.pformat({k: round(v, 4) for k, v in infos.items()}))

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                if self.is_main:
                    print(f"Evaluating at {steps}")

                sval_stats_train = self.eval(env_type="train", seeds=self.args.eval_seeds)
                if self.benchmark == "libero":
                    sval_stats_test = sval_stats_train
                elif self.benchmark == "maniskill":
                    sval_stats_test = self.eval(env_type="test", seeds=self.args.eval_seeds)
                sval_stats = {f"eval/{k}": v for k, v in sval_stats_train.items()}
                sval_stats.update({f"eval/{k}_ood": v for k, v in sval_stats_test.items()})

                if self.is_main:
                    if dist.is_initialized():
                        sval_stats = self.distributed_mean_dict(sval_stats)
                    wandb.log(sval_stats, step=steps)
                    print(f"Eval stats at {steps}:")
                    print(pprint.pformat({k: round(v, 4) for k, v in sval_stats.items()}))

                if self.is_main and episode > max_episodes/10:
                    # 检查是否为最佳模型
                    current_metric = sval_stats.get(f"eval/{self.args.best_model_metric}_ood", 0.0)
                    if current_metric > self.best_metric_value:
                        self.best_metric_value = current_metric
                        self.best_epoch = episode
                        print(f"New best model! {self.args.best_model_metric}: {current_metric:.4f} at epoch {episode}")

                        # 保存最佳模型
                        best_model_path = self.glob_dir / self.args.best_model_dir
                        self.policy.save(best_model_path, epoch=episode,
                                         best_epoch=self.best_epoch, best_metric_value=self.best_metric_value)
                        print(f"Best model saved to {best_model_path}")

                        # 记录到wandb
                        wandb.log({
                            f"best/{self.args.best_model_metric}": current_metric,
                            "best/epoch": episode,
                            "best/steps": steps
                        }, step=steps)
                if dist.is_initialized():
                    dist.barrier()  # 评估后所有进程同步等待

            # checkout last model as reward model
            if self.args.use_reward_update and episode % self.args.update_RM_interval == self.args.update_RM_interval - 1:
                # 将训练模型的参数复制到奖励模型
                for policy_param, reward_param in zip(self.policy.vla.parameters(), self.reward_model.vla.parameters()):
                    # 将训练模型参数从device 0转移到device 1，并复制值
                    reward_param.data = policy_param.data.to(self.reward_device).clone()

                # 确保奖励模型仍然处于评估模式且参数不可训练
                self.reward_model.vla.eval()
                for p in self.reward_model.vla.parameters():
                    p.requires_grad = False

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                if self.is_main:
                    print(f"Saving model at {steps}")
                    save_path = self.glob_dir / f"steps_{episode:0>4d}"
                    self.policy.save(save_path, step=steps, epoch=episode)
                    # 渲染时间超过 NCCL 默认超时（10分钟），就会报 watchdog timeout 错误
                    # 暂时关闭渲染，可以单独启动程序渲染想看的ckpt
                    if self.args.render_video:
                        self.render(epoch=episode, env_type="train")
                        self.render(epoch=episode, env_type="test")
                if dist.is_initialized():
                    dist.barrier()  # 保存/渲染后所有进程同步等待；但是等待上限是10min(NCCL_TIMEOUT)


def main():
    # Parse arguments - tyro will handle config file loading automatically
    args = tyro.cli(CLIArgs)
    args.eval_seeds = parse_eval_seeds_arg(args.eval_seeds)

    if hasattr(args, "config_path") and args.config_path:
        with open(args.config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # 3. 用 config_dict 覆盖 args（命令行参数优先）
        args = replace(args, **{k: v for k, v in config_dict.items() if hasattr(args, k)})

    runner = Runner(args)

    try:
        # Add support for only_eval and only_render for easier testing
        if hasattr(args, 'only_eval') and args.only_eval:
            # Only run evaluation (both train and test sets)
            print("[INFO] Running only evaluation mode...")
            sval_stats_train = runner.eval(env_type="train", seeds=args.eval_seeds)
            if args.benchmark == "libero":
                sval_stats_test = sval_stats_train
            elif args.benchmark == "maniskill":
                sval_stats_test = runner.eval(env_type="test", seeds=args.eval_seeds)
            print("[EVAL/train]", sval_stats_train)
            print("[EVAL/test]", sval_stats_test)
            return

        elif hasattr(args, 'only_render') and args.only_render:
            if args.benchmark == "libero":
                runner.render(epoch=0, env_type="test")
            elif args.benchmark == "maniskill":
                runner.render(epoch=0, env_type=args.env_type)

        else:
            runner.run()

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        # 确保资源清理
        runner.cleanup_async_resources()
        raise
    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        # 确保资源清理
        runner.cleanup_async_resources()
        raise
    else:
        # 正常完成时清理资源
        runner.cleanup_async_resources()


if __name__ == "__main__":
    main()
