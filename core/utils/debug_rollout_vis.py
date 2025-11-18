"""
Debug rollout visualization utilities
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils import visualization
from core.utils.utils import to_numpy, to_list
import numpy as np
from typing import List, Optional


def save_debug_rollout_videos(args, buffer, glob_dir, episode, rank):
    """
    保存rollout调试可视化视频

    Args:
        args: 训练参数
        buffer: replay buffer
        glob_dir: 全局保存目录
        episode: 当前episode编号
        rank: 进程rank
    """
    # 检查是否需要保存debug视频
    if (not args.debug_rollout_vis or
        episode % args.debug_rollout_vis_interval != 0):
        return

    # 创建保存目录，与render保持一致的路径格式
    exp_dir = Path(glob_dir) / "debug" / f"vis_{episode}_train_rank-{rank}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 确定要可视化的环境数量
    num_vis_envs = args.num_envs

    # 从buffer中提取数据
    episode_len = buffer.obs.shape[0] - 1  # buffer.obs包含T+1帧

    for env_idx in range(num_vis_envs):
        # 提取该环境的数据
        images = []
        actions = []
        infos = []

        # 获取instruction
        instruction = buffer.instruction[env_idx]

        # 提取每一步的数据
        success = False
        for step in range(episode_len):
            # 图像 (obs_t)
            obs_img = buffer.obs[step, env_idx]  # shape: (H, W, C)
            images.append(obs_img)

            # 环境信息 (构造info，包含reward、mask等)
            reward = buffer.rewards[step, env_idx, 0]
            mask = buffer.masks[step + 1, env_idx, 0]  # mask[t+1]表示step t后的状态
            done = 1.0 - mask

            # 构造info字典
            info = {
                "reward": float(reward),
                "done": bool(done),
                "step": step,
            }

            # 如果是最后一步，尝试获取成功状态
            success = success or done
            if step == episode_len - 1:
                # 从reward推断成功状态（这里可能需要根据具体环境调整）
                info["success"] = success

            infos.append(info)

        # 添加最后一帧图像 (obs_{T})
        final_obs_img = buffer.obs[episode_len, env_idx]
        images.append(final_obs_img)

        # 确保数据长度正确
        assert len(images) == len(infos) + 1, f"Images: {len(images)}, Infos: {len(infos)}"

        # 在图像上添加信息（如果启用）
        if args.render_info:
            for j in range(len(infos)):
                images[j + 1] = visualization.put_info_on_image(
                    images[j + 1], infos[j],
                    extras=[f"Ins: {instruction}"]
                )

        # 获取成功状态用于文件名
        success = int(infos[-1].get("success", 0))

        # 保存视频，文件名格式与render保持一致
        video_name = f"video_{env_idx:02d}-s_{success}"
        images_to_video(images, str(exp_dir), video_name, fps=10, verbose=False)


def save_buffer_dump(args, buffer, glob_dir, episode, rank, success_array: Optional[np.ndarray] = None, logits_external: Optional[np.ndarray] = None):
    """
    在一轮rollout结束后，将buffer的内容保存为一个npz文件，便于后续分析与可视化。
    文件包含：obs(T+1,N,H,W,3 uint8)、actions(T,N,7 int32)、logprobs(T,N,7 float32)、
    values(T+1,N,1 float32)、rewards(T,N,1 float32)、masks(T+1,N,1 float32)、instruction(N,)、
    以及额外的meta信息（env_id、episode_len、episode、rank）。
    """
    # 若未开启或不在保存间隔，直接返回
    if (not getattr(args, 'buffer_dump', False)) or (episode % getattr(args, 'buffer_dump_interval', 1) != 0):
        return

    # 仅由主进程负责保存，避免多卡重复写
    # 注：train_ms3_ppo.py 中会判断 is_main 再调用本函数

    # 路径：glob/debug/buffer_{episode}_train_rank-{rank}.npz，与 debug 视频保持相似层级
    exp_dir = Path(glob_dir) / "debug"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_path = exp_dir / f"buffer_{episode}_train_rank-{rank}.npz"

    # 打包需要的数据
    # success 数组：如果外部传入，则使用；否则尝试根据 masks 推断（仅供参考）
    if success_array is None:
        # 简单推断：若最后一步 mask 为0，认为该env完成（不代表成功）。严格成功应由env提供
        try:
            last_mask = buffer.masks[-1].reshape(-1)  # [N]
            success_array = (1.0 - last_mask).astype(bool)
        except Exception:
            success_array = None

    # 准备保存数据
    save_data = {
        'actions': buffer.actions,
        'action_log_probs': buffer.action_log_probs,
        'value_preds': buffer.value_preds,
        'rewards': buffer.rewards,
        'advantages': buffer.advantages,
        'masks': buffer.masks,
        'instruction': np.array(buffer.instruction, dtype=object),
        'meta_env_id': args.env_id,
        'meta_episode_len': args.episode_len,
        'meta_episode': episode,
        'meta_rank': rank,
        'success': success_array,
    }


    # 新逻辑：优先使用外部传入的 logits_tensors（若提供），否则尝试从 buffer 读取（兼容旧逻辑）
    try:
        T = buffer.actions.shape[0]
        if logits_external is not None:
            logits_arr = logits_external[:T]
            save_data['logits_tensors'] = logits_arr
            valid_counts = (buffer.masks[1:, :, 0] == 1).sum(axis=1).astype(np.int32)
            save_data['logits_original_env_counts'] = valid_counts
            save_data['logits_max_env'] = buffer.actions.shape[1]
            print(f"[buffer dump] 成功保存外部 logits_tensors（形状：{logits_arr.shape}）")
            print(f"[buffer dump] 每步有效环境数：{valid_counts.tolist()}，最大环境数：{int(save_data['logits_max_env'])}")
        elif hasattr(buffer, 'logits_tensors') and getattr(buffer, 'logits_tensors') is not None:
            logits_arr = buffer.logits_tensors[:T]
            save_data['logits_tensors'] = logits_arr
            valid_counts = (buffer.masks[1:, :, 0] == 1).sum(axis=1).astype(np.int32)
            save_data['logits_original_env_counts'] = valid_counts
            save_data['logits_max_env'] = buffer.actions.shape[1]
            print(f"[buffer dump] 成功保存 logits_tensors（形状：{logits_arr.shape}）")
            print(f"[buffer dump] 每步有效环境数：{valid_counts.tolist()}，最大环境数：{int(save_data['logits_max_env'])}")
    except Exception as e:
        print(f"[buffer dump] Warning: Failed to save logits_tensors: {e}")

    np.savez_compressed(save_path, **save_data)

    print(f"[buffer dump] saved to: {save_path}")
