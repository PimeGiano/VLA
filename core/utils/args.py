# config.py
from dataclasses import dataclass, field
from typing import Annotated, Union, Optional, Dict, Any
import tyro


# ======== 1. 父类：放公共参数 ========
@dataclass(kw_only=True)  # kw_only=True 只支持“关键字参数”，这样参数顺序就没影响，所有参数都可以随意有无默认值
class BaseArgs:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""
    env_type: Optional[str] = None

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-test"

    async_enabled: bool = False

    # env
    num_envs: int = 64
    episode_len: int = 80
    use_same_init: bool = False

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 10
    interval_save: int = 10

    # buffer
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    update_valid_envs: bool = False

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

    # 学习率调度器参数
    warmup_steps: Optional[int] = None  # 直接指定 warmup 步数，如果为 None 则使用 warmup_ratio 计算
    warmup_ratio: float = 0.03  # warmup 步数占总训练步数的比例
    max_train_steps: Optional[int] = None  # 总训练步数，如果为 None 则根据其他参数自动计算

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1

    use_global_ppo: bool = False

    ppo_clip: float = 0.2

    policy_coef: Union[float, list[tuple[float, float]]] = 1.0  # 策略loss系数，支持常数或[(t,v)]分段线性
    value_coef: Union[float, list[tuple[float, float]]] = 1.0   # 价值损失系数，支持常数或[(t,v)]分段线性
    alg_entropy_coef: Union[float, list[tuple[float, float]]] = 0.0  # 熵损失系数，支持常数或[(t,v)]分段线性
    # distribution_loss_coef: Union[float, list[tuple[float, float]]] = 0.0  # 分布loss码，支持常数或[(t,v)]分段线性
    distribution_loss_coef: Union[float, list[tuple[float, float]], str] = field(
        default_factory=lambda: [(0.0, 1.0), (0.3, 1.0), (1.0, 0.1)]
    )  # 分布loss码，支持常数或[(t,v)]分段线性

    # distribution loss options
    dist_tau: float = 0.0              # 温度 τ：τ<=0 使用 argmax 中心；τ>0 使用 soft-argmax；越大越平滑
    dist_trunc_k: float = 3.0          # 截断窗口系数；<=0 或 inf 关闭截断，在全域上计算 KL
    dist_topk: Optional[int] = None    # 概率质量 top-k 选择；None 表示不启用
    kernel_sigma: float = 1.0          # 高斯核带宽 sigma
    dist_loss_type: str = "simplest"   # simplest / gaussian_shape / gaussian_kernel
    dist_constrained_dims: int = 6     # 参与分布约束的前几个 action 维度
    dist_max_chunk_num: Optional[int] = 1  # 仅对前多少个 chunk 施加分布损失，None 表示全部
    dist_min_sigma: float = 1e-3       # σ 的下限
    dist_fixed_sigma: Optional[float] = None  # 固定 σ；若为空则根据分布估计
    dist_peak_window_bins: Optional[int] = None  # 高斯核损失下，以 argmax 为中心的窗口大小
    dist_eps: float = 1e-6
    dist_action_dim: Optional[int] = None  # compute_diss_loss 中的 action_dim，None 表示自动推断

    # other
    config_path: str = ""
    """Path to YAML config file for loading arguments (command-line args override)."""
    wandb: bool = False
    only_render: bool = False
    render_video: bool = True
    render_info: bool = True
    resume: bool = False  # wandb 的历史数据表是追加式的,所以resume时回退的那些step在上传时会被忽略,无法覆盖写入

    only_eval: bool = False
    """If set, only run evaluation (no training)."""
    only_render: bool = False
    """If set, only run rendering (no training)."""

    eval_seeds: str = "0,1,2"
    """评估时使用的全局偏移种子列表；长度决定重复评估次数。"""

    # debug visualization during rollout
    debug_rollout_vis: bool = False
    """If set, enable visualization during rollout for debugging purposes."""
    debug_rollout_vis_interval: int = 1
    """Interval (in episodes) for rollout visualization when debug_rollout_vis is enabled."""

    # buffer dump
    buffer_dump: bool = False
    """是否在每轮rollout结束后保存buffer数据(npz)。"""
    buffer_dump_interval: int = 1
    """保存buffer的间隔（按episode计）。"""

    # best model tracking
    best_model_metric: str = "success"  # 用于判断最佳模型的指标
    best_model_dir: str = "best_model"  # 最佳模型保存目录

    use_endoRM: bool = False # 使用内生奖励
    use_avg_endo: bool = False # 使用平均内生奖励
    with_VQ: bool = True # V_Q_1
    endo_reward_reg_coef: float = 1.0  # 内生奖励正则项系数 1.0表示不变 越小越尖锐 越大越平均(随机)
    endo_reward_scale: float = 500.0  # 内生奖励缩放系数
    reward_window_size: int = 10 # reward chunking的长度
    reward_gpu_rank: str = "3"  # 每个训练GPU对应的reward模型GPU在可见GPU列表中的rank，格式如"4,4,5,5"
    reward_model_path: Optional[str] = ""  # reward model加载路径，若为空则与vla_path一致
    reward_vla_load_path: Optional[str] = ""  # reward model加载路径，若为空则与vla_path一致
    self_reward: bool = False  # 使用训练模型自身计算内生奖励，而非独立的reward模型

    # 更新 reward model
    use_reward_update: bool = False # 使用reward model更新
    update_RM_interval: int = 20 # 更新 reward model的轮次

# ======== 2. 子类：各自补专有参数（暂时为空） ========
@dataclass(kw_only=True)
class LIBEROArgs(BaseArgs):
    _name_: str = "libero"

    benchmark: str = "libero"

    # update_valid_envs = False
    update_valid_envs: bool = True

    # TODO
    # num_trials_per_task: int = 50
    num_trials_per_task: int = 1

    # vla_unnorm_key: str = "libero_spatial_no_noops"
    vla_unnorm_key: str = "libero_spatial"
    # vla_unnorm_key: str = "bridge_orig"
    model_family: str = "openvla"

    # vla_path: str = "./weights/moojink/openvla-7b-oft-finetuned-libero-spatial"
    # vla_path: str = "./weights/openvla/openvla-7b"
    vla_path: str = "./weights/openvla/openvla-7b-finetuned-libero-spatial"

    # env
    img_width: int = 256
    img_height: int = 256
    num_steps_wait: int = 10

    # train
    num_envs: int = 20
    episode_len: int = 300
    use_same_init: bool = False

    vla_lr: float = 1e-5
    vla_vhlr: float = 3e-4
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    alg_ppo_epoch: int = 1

    task_suite_name_train: Optional[Dict[str, Any]] = None
    task_suite_name_test: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True)
class ManiSkillArgs(BaseArgs):
    _name_: str = "maniskill"

    benchmark: str = "maniskill"

    vla_path: str = "./weights/openvla-7b-rlvla-warmup"

    img_width: int = 640
    img_height: int = 480

    update_valid_envs: bool = False


# ======== 3. 汇总：Union 即可 ========
# tyro 在解析 CLI 时若收到一个 Union，会把其中每个成员类当作 子命令（sub-command）
# 命令行用户需要在首个位置写上 train 或 eval，tyro 才知道接下来该用哪一套字段进行解析
CLIArgs = Union[LIBEROArgs, ManiSkillArgs]
