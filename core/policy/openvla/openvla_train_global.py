# -*- coding: utf-8 -*-
"""
openvla_train_global.py

方案一（全局汇总再分配）的独立实现：
- 在每个 rank 完成 buffer 的 returns/advantages 计算后，收集本地有效样本；
- 使用 distributed.gather_object 在 rank0 汇总；rank0 进行拼接、必要的重复采样补齐至均匀切分；
- 使用 distributed.scatter_object_list 将均匀切分后的 shard 分发到各个 rank；
- 各 rank 基于本地 shard 构造本地 data generator，按原 train_ppo_step 执行训练；

注意：
- 为简化实现与最小改动，本实现基于 Python 对象收集（gather_object / scatter_object_list），
  在高分辨率图像和大批量情况下可能带来较高的通信/序列化开销；如需更优性能，可按需改为张量分片与分块通信。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

# 复用现有的 Policy 与基础 PPO 步进逻辑
from core.policy.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO


@dataclass
class _Shard:
    """封装一个数据分片（numpy/py对象集合，用于对象通信）"""
    obs: np.ndarray  # [M, ...] uint8
    actions: np.ndarray  # [M, act_dim]
    value_preds: np.ndarray  # [M, 1]
    returns: np.ndarray  # [M, 1]
    masks: np.ndarray  # [M, 1]
    old_action_logits: np.ndarray  # [M, act_dim]
    advantages: np.ndarray  # [M, 1]
    instruct: List[str]  # len M


class OpenVLAPPOGlobal(OpenVLAPPO):
    """PPO（全局汇总再分配版本）"""

    def _flatten_buffer(self, buffer) -> Dict[str, Any]:
        """将 buffer 展平到 [T*N, ...] 级别，返回 numpy/py 对象集合。"""
        episode_length, n_rollout_threads = buffer.rewards.shape[:2]
        obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:])
        actions = buffer.actions.reshape(-1, buffer.actions.shape[-1])
        value_preds = buffer.value_preds[:-1].reshape(-1, 1)
        returns = buffer.returns.reshape(-1, 1)
        masks = buffer.masks[:-1].reshape(-1, 1)
        old_action_logits = buffer.action_log_probs.reshape(-1, buffer.action_log_probs.shape[-1])
        advantages = buffer.advantages.reshape(-1, 1)

        # instruct 展平：index % n_rollout_threads
        flat_size = episode_length * n_rollout_threads
        instruct_indices = np.arange(flat_size) % n_rollout_threads
        instruct = [buffer.instruction[i] for i in instruct_indices]

        return dict(
            obs=obs,
            actions=actions,
            value_preds=value_preds,
            returns=returns,
            masks=masks,
            old_action_logits=old_action_logits,
            advantages=advantages,
            instruct=instruct,
            episode_length=episode_length,
            n_rollout_threads=n_rollout_threads,
        )

    def _filter_valid(self, flat: Dict[str, Any], use_valid: bool) -> Dict[str, Any]:
        """按 masks==1 过滤有效样本；不使用时返回全部。"""
        if not use_valid:
            return {k: v for k, v in flat.items() if k in (
                'obs', 'actions', 'value_preds', 'returns', 'masks', 'old_action_logits', 'advantages', 'instruct')}

        valid_mask = (flat['masks'] == 1).reshape(-1)
        valid_idx = np.where(valid_mask)[0]
        if valid_idx.size == 0:
            # 返回空集，后续 rank0 会做补齐
            return {k: (v if k == 'instruct' else v[:0]) for k, v in flat.items() if k in (
                'obs', 'actions', 'value_preds', 'returns', 'masks', 'old_action_logits', 'advantages', 'instruct')}

        out = {
            'obs': flat['obs'][valid_idx],
            'actions': flat['actions'][valid_idx],
            'value_preds': flat['value_preds'][valid_idx],
            'returns': flat['returns'][valid_idx],
            'masks': flat['masks'][valid_idx],
            'old_action_logits': flat['old_action_logits'][valid_idx],
            'advantages': flat['advantages'][valid_idx],
            'instruct': [flat['instruct'][i] for i in valid_idx.tolist()],
        }
        return out

    def _concat_and_shard(self, parts: List[Dict[str, Any]], world_size: int) -> List[_Shard]:
        """在 rank0 上拼接所有 rank 的数据，并重复采样补齐到可均匀切分，返回每个 rank 的 shard 列表。"""
        # 拼接
        def cat(arrs, axis=0):
            if len(arrs) == 0:
                return None
            sizes = [a.shape[0] for a in arrs]
            if sum(sizes) == 0:
                # 全空
                return arrs[0]
            return np.concatenate(arrs, axis=axis)

        obs = cat([p['obs'] for p in parts])
        actions = cat([p['actions'] for p in parts])
        value_preds = cat([p['value_preds'] for p in parts])
        returns = cat([p['returns'] for p in parts])
        masks = cat([p['masks'] for p in parts])
        old_action_logits = cat([p['old_action_logits'] for p in parts])
        advantages = cat([p['advantages'] for p in parts])
        instruct = sum([p['instruct'] for p in parts], [])

        total = 0 if obs is None else obs.shape[0]
        if total == 0:
            # 所有 rank 都无有效样本，构造 world_size 份空 shard
            return [
                _Shard(obs[:0], actions[:0], value_preds[:0], returns[:0], masks[:0], old_action_logits[:0], advantages[:0], [])
                for _ in range(world_size)
            ]

        # 计算每 rank 目标大小（均匀，向上取整）
        per_rank = int(np.ceil(total / world_size))
        total_needed = per_rank * world_size
        if total_needed > total:
            shortage = total_needed - total
            # 重复采样补齐
            idx_all = np.arange(total)
            sup = np.random.choice(idx_all, size=shortage, replace=True)
            sel = np.concatenate([idx_all, sup])
            obs, actions = obs[sel], actions[sel]
            value_preds, returns = value_preds[sel], returns[sel]
            masks, old_action_logits = masks[sel], old_action_logits[sel]
            advantages = advantages[sel]
            instruct = [instruct[i] for i in sel.tolist()]
            total = total_needed

        # 随机打乱再切分
        perm = np.random.permutation(total)
        obs, actions = obs[perm], actions[perm]
        value_preds, returns = value_preds[perm], returns[perm]
        masks, old_action_logits = masks[perm], old_action_logits[perm]
        advantages = advantages[perm]
        instruct = [instruct[i] for i in perm.tolist()]

        shards: List[_Shard] = []
        for r in range(world_size):
            s, e = r * per_rank, (r + 1) * per_rank
            shards.append(_Shard(
                obs[s:e], actions[s:e], value_preds[s:e], returns[s:e], masks[s:e], old_action_logits[s:e], advantages[s:e], instruct[s:e]
            ))
        return shards

    def train_ppo(self, buffer, progress=None):
        train_info = defaultdict(lambda: [])
        train_steps = 0

        # === 时间统计初始化 ===
        import time
        t0 = time.time()

        # 计算 returns/advantages
        buffer.compute_returns_ppo()
        use_valid = getattr(buffer, 'update_valid_envs', False)

        # 准备本地展平数据并按需过滤
        flat = self._flatten_buffer(buffer)
        local_part = self._filter_valid(flat, use_valid)

        t_after_flatten = time.time()

        # 分布式收集与分发
        gather_time = shard_time = scatter_time = 0.0
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # gather 到 rank0
            tg0 = time.time()
            if rank == 0:
                gathered: List[Dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore
                dist.gather_object(local_part, object_gather_list=gathered, dst=0)
                tg1 = time.time()
                # 在 rank0 拼接并切分
                ts0 = time.time()
                shard_list = self._concat_and_shard(gathered, world_size)
                ts1 = time.time()
            else:
                dist.gather_object(local_part, object_gather_list=None, dst=0)
                tg1 = time.time()
                shard_list = None
                ts0 = ts1 = tg1
            gather_time = tg1 - tg0
            shard_time = ts1 - ts0

            # scatter 到各 rank
            tc0 = time.time()
            recv_list = [None]
            if rank == 0:
                dist.scatter_object_list(recv_list, scatter_object_input_list=shard_list, src=0)
            else:
                dist.scatter_object_list(recv_list, scatter_object_input_list=[], src=0)
            tc1 = time.time()
            scatter_time = tc1 - tc0

            shard: _Shard = recv_list[0]  # type: ignore
        else:
            # 单卡：本地数据直接作为 shard，并做一次均匀性处理（等同 1 份）
            shard = self._concat_and_shard([local_part], 1)[0]

        t_after_scatter = time.time()

        # 基于分片构造本地 mini-batch 训练
        # 选择 mini-batch 大小
        mb_size = buffer.buffer_minibatch if buffer.buffer_minibatch > 0 else shard.obs.shape[0]
        total = shard.obs.shape[0]
        num_mini_batch = int(np.ceil(total / max(1, mb_size)))

        def _iter_batches():
            idx = np.random.permutation(total)
            for i in range(num_mini_batch):
                s, e = i * mb_size, min((i + 1) * mb_size, total)
                sel = idx[s:e]
                # 构造与原接口一致的 batch 元组
                obs_batch = shard.obs[sel]
                actions_batch = shard.actions[sel]
                value_preds_batch = shard.value_preds[sel]
                return_batch = shard.returns[sel]
                masks_batch = shard.masks[sel]
                old_action_logits_batch = shard.old_action_logits[sel]
                adv_targ = shard.advantages[sel]
                instruct_batch = [shard.instruct[j] for j in sel.tolist()]
                yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                       old_action_logits_batch, adv_targ)

        iterator = _iter_batches()
        loop = enumerate(iterator)

        for idx, batch in tqdm(
                loop,
                total=num_mini_batch,
                desc=f"train [rank {self.rank}]",
                position=self.rank + 1,
                leave=(self.rank == 0)):
            # 训练进度（0-1）：优先使用来自外层 episode 的全局进度；否则保持默认行为
            info = self.train_ppo_step(idx, num_mini_batch, batch, progress=progress)
            for k, v in info.items():
                train_info[k].append(v)
            train_steps += batch[-1].shape[0]

        # 学习率调度器步进（与原逻辑一致：每 epoch 一次）
        if hasattr(self.policy, 'vh_scheduler') and self.policy.vh_scheduler is not None:
            self.policy.vh_scheduler.step()
        if hasattr(self.policy, 'vla_scheduler') and self.policy.vla_scheduler is not None:
            self.policy.vla_scheduler.step()

        final_info = {}
        for k, v in train_info.items():
            final_info[k] = np.mean(v)
        final_info["train_steps"] = train_steps

        # === 时间统计输出 ===
        t1 = time.time()
        final_info["dist/gather_s"] = gather_time
        final_info["dist/concat_shard_s"] = shard_time
        final_info["dist/scatter_s"] = scatter_time
        final_info["time/flatten_s"] = (t_after_flatten - t0)
        final_info["time/post_scatter_s"] = (t_after_scatter - t_after_flatten)
        final_info["time/train_s"] = (t1 - t_after_scatter)
        final_info["time/total_s"] = (t1 - t0)

        return final_info

