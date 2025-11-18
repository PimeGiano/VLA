import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated
import torch
import torch.distributed as dist
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

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
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
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


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-test"

    # env
    num_envs: int = 64
    episode_len: int = 80
    use_same_init: bool = False

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 10
    interval_save: int = 40

    # buffer
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    # vla
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    vla_lora_rank: int = 32

    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.0

    # other
    wandb: bool = True
    only_render: bool = False
    render_video: bool = True
    render_info: bool = True
    resume: bool = False  # wandb 的历史数据表是追加式的,所以resume时回退的那些step在上传时会被忽略,无法覆盖写入

    # best model tracking
    best_model_metric: str = "success"  # 用于判断最佳模型的指标
    best_model_dir: str = "best_model"  # 最佳模型保存目录



class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # 分布式初始化
        dist_info = init_distributed()
        if len(dist_info) == 3:
            self.rank, self.world_size, self.local_rank = dist_info
        else:
            self.rank, self.world_size = dist_info
            self.local_rank = 0
        self.is_main = is_main_process()

        # set seed (不同rank不同seed)
        np.random.seed(self.args.seed + self.rank)
        random.seed(self.args.seed + self.rank)
        torch.manual_seed(self.args.seed + self.rank)

        # 只在主进程初始化wandb和保存目录
        if self.is_main:
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
                    mode="online" if self.args.wandb else "offline",
                )
            wandb.run.tags = wandb.run.tags + (wandb.run.id,)
            self.save_dir = Path(wandb.run.dir)
            self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
            self.glob_dir.mkdir(parents=True, exist_ok=True)
            yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))
        else:
            # 非主进程不访问wandb.run.dir，直接用临时目录
            self.save_dir = None
            self.glob_dir = None

        # policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        # device_id = 0
        # device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        # self.device = torch.device("cuda:" + str(device_id))
        # self.policy = OpenVLAPolicy(all_args, device_id_other)

        self.device = torch.device(f"cuda:{self.local_rank}")
        self.policy = OpenVLAPolicy(all_args, self.local_rank)
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
                find_unused_parameters=True,
                gradient_as_bucket_view=True
            )

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
        self.env = SimlerWrapper(self.args, unnorm_state, extra_seed=self.local_rank, device_id=self.local_rank)

        # buffer
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
        )
        minibatch_count = self.buffer.get_minibatch_count()
        if self.is_main:
            print(f"Buffer minibatch count: {minibatch_count}")

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)

        values = torch.cat(values, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0).to(device=self.device)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.device)

        return values, actions, logprobs

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)

        return value, action, logprob

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)

        obs_img = obs_img.cpu().numpy()
        actions = actions.to(torch.int32).cpu().numpy()
        logprob = logprob.to(torch.float32).cpu().numpy()
        value_preds = value_preds.to(torch.float32).cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()

        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks)

    def compute_endup(self):
        # 为什么这里要prep_rollout？ 似乎是冗余的添加，为了确保不会出错 比如def collect也有
        self.policy.prep_rollout()

        obs_image = torch.tensor(self.buffer.obs[-1]).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        next_value = next_value.to(torch.float32).cpu().numpy()

        self.buffer.endup(next_value)

    def train(self):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer)
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
            values = torch.tensor([float(info[k]) for k in keys], device=self.device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= dist.get_world_size()
            # print(f"[rank {self.rank}] after all_reduce")
            return dict(zip(keys, values.cpu().numpy()))
        except Exception as e:
            print(f"[rank {self.rank}] all_reduce failed: {e}")
            import sys
            sys.exit(1)

    @torch.no_grad()
    def eval(self, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])

        obs_img, instruction, info = self.env.reset(obj_set=obj_set)

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            # info
            # print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        # print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        # print(f"")

        return env_stats

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        # 只在主进程执行渲染和保存
        if not self.is_main:
            return {}
            
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]

        obs_img, instruction, info = self.env.reset(obj_set)
        print("instruction[:3]:", instruction[:3])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                post_action = self.env._process_action(action)
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[f"Ins: {instruction[i]}"]
                    )

            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}",
                            fps=10, verbose=False)

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() for k, v in env_stats.items()}
        save_stats["instruction"] = {idx: ins for idx, ins in enumerate(instruction)}
        save_stats["last_info"] = last_info

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

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

            obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
            self.buffer.warmup(obs_img.cpu().numpy(), instruction)

            # 内层rollout进度条（每个rank一行，只有rank 0 leave=True，其余leave=False）
            for _ in tqdm(
                    range(self.args.episode_len),
                    desc=f"rollout [rank {self.rank}]",
                    position=self.rank + 1,
                    leave=(self.rank == 0)):
                value, action, logprob = self.collect()
                obs_img, reward, done, env_info = self.env.step(action)

                data = (obs_img, action, logprob, value, reward, done)
                self.insert(data)

                # info
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v

            # steps
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs
            print(f"[rank {self.rank}] " + pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))

            # train and process infos
            # 最后一步的 value（即 episode 结束时的 value），在采样时还没算出来，需要在 episode 结束后单独补上
            self.compute_endup()
            del value, action, logprob, obs_img, reward, done
            # CPU内存释放
            gc.collect()
            # GPU内存释放
            torch.cuda.empty_cache()

            # train
            infos = self.train()

            # env_infos，后面会调用self.distributed_mean_dict同步到所有进程
            env_stats = {f"env/{k}": np.mean(v) for k, v in env_infos.items()}
            infos.update(env_stats)

            # 同步所有进程的infos，用于wandb记录
            if dist.is_initialized():
                infos = self.distributed_mean_dict(infos)
            if self.is_main:
                wandb.log(infos, step=steps)
                elapsed_time = time.time() - ep_time
                print(f"{self.args.name}: ep {episode:0>4d} | steps {steps} | e {elapsed_time:.2f}s")
                print(pprint.pformat({k: round(v, 4) for k, v in infos.items()}))

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                if self.is_main:
                    print(f"Evaluating at {steps}")
                
                sval_stats_train = self.eval(obj_set="train")
                sval_stats_test = self.eval(obj_set="test")
                sval_stats = {f"eval/{k}": v for k, v in sval_stats_train.items()}
                sval_stats.update({f"eval/{k}_ood": v for k, v in sval_stats_test.items()})

                if self.is_main:
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

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                if self.is_main:
                    print(f"Saving model at {steps}")
                    save_path = self.glob_dir / f"steps_{episode:0>4d}"
                    self.policy.save(save_path, epoch=episode)
                    # 渲染时间超过 NCCL 默认超时（10分钟），就会报 watchdog timeout 错误
                    # 暂时关闭渲染，可以单独启动程序渲染想看的ckpt
                    if self.args.render_video:
                        self.render(epoch=episode, obj_set="train")
                        self.render(epoch=episode, obj_set="test")
                if dist.is_initialized():
                    dist.barrier()  # 保存/渲染后所有进程同步等待；但是等待上限是10min(NCCL_TIMEOUT)


def main():
    args = tyro.cli(Args)
    runner = Runner(args)

    if args.only_render:
        ll = [
            "PutOnPlateInScene25VisionImage-v1",
            "PutOnPlateInScene25VisionTexture03-v1",
            "PutOnPlateInScene25VisionTexture05-v1",
            "PutOnPlateInScene25VisionWhole03-v1",
            "PutOnPlateInScene25VisionWhole05-v1",

            "PutOnPlateInScene25Instruct-v1",
            "PutOnPlateInScene25Plate-v1",
            "PutOnPlateInScene25Position-v1",
            "PutOnPlateInScene25EEPose-v1",
            "PutOnPlateInScene25PositionChange-v1",
            "PutOnPlateInScene25PositionChangeTo-v1"
        ]
        if args.env_id not in ll:
            runner.render(epoch=0, obj_set="train")
        runner.render(epoch=0, obj_set="test")
    else:
        runner.run()


if __name__ == "__main__":
    main()
