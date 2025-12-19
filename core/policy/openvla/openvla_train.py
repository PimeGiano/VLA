import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, BatchFeature, get_constant_schedule_with_warmup

from core.utils.coef_schedules import eval_schedule, parse_schedule_def  # 解析来自CLI的字符串/列表格式
# from core_rlinf.algorithms.losses import compute_diss_loss
from core.policy.losses import compute_diss_loss

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class OpenVLAPolicy:
    def __init__(self, all_args, device: torch.device):
        self.args = all_args
        self.device = device
        # Tensor Precision Device Vector
        self.tpdv = dict(device=self.device, dtype=torch.bfloat16)
        # Value Network Precision Device Vector
        self.tpdv_vn = dict(device=self.device, dtype=torch.float32)

        self.action_scale = 1.0
        self.start_step = 0  # for resume
        self.start_epoch = 0  # for resume
        self.best_epoch = -1
        self.best_metric_value = -1.0

        # openvla: register
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        self.image_processor = PrismaticImageProcessor.from_pretrained(self.args.vla_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vla_path, trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        # self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device,
            vh_mode="a0",
        )
        self.vla.args = self.args

        # openvla: lora
        if not self.args.vla_load_path:
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "proj", "qkv", "fc1", "fc2",  # vision
                    "q", "kv", "fc3",  # project
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",  # llm
                ],
                init_lora_weights="gaussian"
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            # PEFT/LoRA的from_pretrained 只会加载LoRA adapter的权重，不会加载自定义的value_head、optimizer等
            # 所以下面还有一次加载
            self.vla = PeftModel.from_pretrained(self.vla, self.args.vla_load_path, is_trainable=True)
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        # set value head trainable
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        self.vla.print_trainable_parameters()

        # openvla: optimizer
        # for value head trainable parameters
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        # for value head
        self.vla_optimizer = None
        self._setup_optimizer()

        if self.args.vla_load_path:
            # 自定义的training_state.pt里保存了value_head和optimizer状态，所以需要单独再load一次
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

                if "vh" in training_state:
                    self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
                else:
                    print("Warning: value_head state not found in training_state")

                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
                self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

                # 加载学习率调度器状态
                if 'vh_scheduler' in training_state and hasattr(self, 'vh_scheduler'):
                    self.vh_scheduler.load_state_dict(training_state['vh_scheduler'])
                if 'vla_scheduler' in training_state and hasattr(self, 'vla_scheduler'):
                    self.vla_scheduler.load_state_dict(training_state['vla_scheduler'])

                print(f"Optimizer load: {self.args.vla_load_path}")

                if self.args.resume:
                    self.start_step = training_state.get("step", None)
                    self.start_epoch = training_state.get("epoch", -1) + 1
                    self.best_epoch = training_state.get("best_epoch", -1)
                    self.best_metric_value = training_state.get("best_metric_value", -1.0)
            else:
                print(f"Warning: training_state not found in {training_state_path}")

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)

        # 强化学习中不需要根据GPU数量缩放学习率
        # 因为每个GPU处理独立的环境，梯度已经通过DDP自动平均
        import torch.distributed as dist

        # 使用原始学习率，不进行缩放
        vla_lr = self.args.vla_lr
        vh_lr = self.args.vla_vhlr

        self.vh_optimizer = AdamW(self.params_vh, lr=vh_lr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=vla_lr, betas=betas)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            print(f"[DDP] Using original LR (no scaling): vla={vla_lr}, vh={vh_lr}")
            print(f"[DDP] World size: {world_size}, effective batch size per GPU: {self.args.num_envs}")
            print(f"[DDP] Total effective batch size: {self.args.num_envs * world_size}")

            # 可选：根据world_size调整梯度累积步数以保持一致的优化器更新频率
            # 这里注释掉，如果需要可以启用
            # original_grad_accum = self.args.alg_gradient_accum
            # self.args.alg_gradient_accum = max(1, original_grad_accum // world_size)
            # print(f"[DDP] Adjusted gradient accumulation: {original_grad_accum} -> {self.args.alg_gradient_accum}")

        # 设置学习率调度器
        if self.args.warmup_ratio > 0:
            self._setup_lr_schedulers()

    def _setup_lr_schedulers(self):
        """设置学习率调度器，支持 warmup 功能"""
        # 计算总训练步数
        total_training_steps = self._calculate_total_training_steps()

        # 计算 warmup 步数
        if hasattr(self.args, "warmup_steps") and self.args.warmup_steps is not None:
            warmup_steps = self.args.warmup_steps
        else:
            warmup_ratio = getattr(self.args, "warmup_ratio", 0.03)
            warmup_steps = int(total_training_steps * warmup_ratio)

        # 创建学习率调度器
        # get_constant_schedule_with_warmup：线性升温后保持常数学习率
        # get_linear_schedule_with_warmup：线性升温后线性衰减到 0
        # get_cosine_schedule_with_warmup：线性升温后余弦衰减
        # get_polynomial_decay_schedule_with_warmup：线性升温后按多项式衰减到指定 end lr
        self.vla_scheduler = get_constant_schedule_with_warmup(
            self.vla_optimizer,
            num_warmup_steps=warmup_steps,
            # num_training_steps=total_training_steps,
        )
        self.vh_scheduler = get_constant_schedule_with_warmup(
            self.vh_optimizer,
            num_warmup_steps=warmup_steps,
            # num_training_steps=total_training_steps,
        )

        print(f"[LR Scheduler] Total training steps: {total_training_steps}")
        print(f"[LR Scheduler] Warmup steps: {warmup_steps} ({warmup_steps/total_training_steps*100:.1f}%)")

    def _calculate_total_training_steps(self):
        """计算总训练步数"""
        # 如果直接指定了 max_train_steps，使用该值
        if hasattr(self.args, "max_train_steps") and self.args.max_train_steps is not None:
            return self.args.max_train_steps

        # 否则根据训练配置计算
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        total_steps = max_episodes * self.args.alg_ppo_epoch

        return total_steps

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8

        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(**self.tpdv)

        # prompt
        if action is None:
            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: "
                           for t in task_description]
        else:
            assert isinstance(action, torch.Tensor)
            # action = action.cpu().numpy() # [B, dim]
            action_str = self.tokenizer.batch_decode(action)

            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: {a}</s>"
                           for t, a in zip(task_description, action_str)]

        inputs = self.processor(task_prompt, images, padding=True)
        inputs = inputs.to(**self.tpdv)

        if action is not None:
            inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)
        features = self._preprocess_obs(x)

        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        values, action, logprobs, logits_tensor = vla_model.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
            # alpha = self.args.endo_reward_reg_coef
        )

        assert len(values.shape) == 2 and values.shape[1] == 1
        assert len(action.shape) == 2 and action.shape[0] == values.shape[0]
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, action, logprobs, logits_tensor

    def get_endo_reward(self, x: dict, action, without_preproc=False) -> torch.Tensor:
        # reference to evaluate_actions
        # soft softmax 系数
        coef = self.args.endo_reward_reg_coef
        if without_preproc:
            # TODO
            pass
        else:
            features = self._preprocess_obs(x, action)

        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        endo_reward = vla_model.get_endo_reward(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            coef=coef,
            with_VQ=self.args.with_VQ
        )

        assert len(endo_reward.shape) == 2 and endo_reward.shape[1] == 1

        return endo_reward

    def get_endoreward_bk(self, x: dict, generated_ids) -> torch.Tensor:
        # soft softmax 系数
        temperature = 1.0  # not used
        do_sample = (temperature != 0.0)

        features = self._preprocess_obs(x)
        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        input_ids = features['input_ids'].to(self.device)
        attention_mask = features['attention_mask'].to(self.device)
        pixel_values = features['pixel_values'].to(self.device)

        batch_size = input_ids.shape[0]
        # 创建一个形状为 [batch_size, 1] 的全零张量，并确保设备正确
        endo_rewards = torch.zeros(batch_size, 1, device=input_ids.device)

        for i in range(generated_ids.shape[1]):

            endo_reward = vla_model.predict_endoreward_batch(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                actions = generated_ids[:,i:i+1],
                do_sample=do_sample,
                temperature=temperature
            )

            endo_rewards += endo_reward

            input_ids = torch.cat([input_ids, generated_ids[:, i:i+1]], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(generated_ids[:, i:i+1])], dim=1)


        assert len(endo_rewards.shape) == 2 and endo_rewards.shape[1] == 1

        return endo_rewards

    def get_action_temp(self, x: dict, do_sample, temperature, num_beams) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x)

        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        _, action, logprobs = vla_model.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
        )

        assert len(action.shape) == 2
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return action, logprobs

    def get_value(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        value = vla_model.get_value(**features)

        assert len(value.shape) == 2 and value.shape[1] == 1

        return value

    def get_hidden(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        hs = vla_model.get_hidden(**features)

        return hs

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x, action)

        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        logprobs, entropy, values, logits_tensor = vla_model.evaluate_action(
            **features,
            unnorm_key=self.args.vla_unnorm_key
        )

        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        assert len(values.shape) == 2 and values.shape[1] == 1

        # 返回4元组（含 logits_tensor），与 get_action 返回保持一致，便于上层按需使用
        return logprobs, entropy, values, logits_tensor

    def prep_rollout(self):
        # DDP本身就有eval()方法
        self.vla.eval()

    def prep_training(self):
        self.vla.train()

    def save(self, path: Path, step: int = None, epoch: int = None, best_epoch: int = -1, best_metric_value: float = -1.0):
        path.mkdir(parents=True, exist_ok=True)
        vla_model = self.vla.module if hasattr(self.vla, "module") else self.vla
        vla_model.save_pretrained(str(path))
        training_state = {
            "vh": vla_model.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric_value": best_metric_value,
        }

        # 保存学习率调度器状态
        if hasattr(self, 'vh_scheduler') and self.vh_scheduler is not None:
            training_state["vh_scheduler"] = self.vh_scheduler.state_dict()
        if hasattr(self, 'vla_scheduler') and self.vla_scheduler is not None:
            training_state["vla_scheduler"] = self.vla_scheduler.state_dict()
        torch.save(training_state, path / "training_state.pt")

        json.dump(vla_model.base_model.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path):
        # Note: 没看到哪里用到，所以暂时未针对DDP做修改
        del self.vla
        torch.cuda.empty_cache()

        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        training_state_path = path / "training_state.pt"
        training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

        # 加载学习率调度器状态
        if 'vh_scheduler' in training_state and hasattr(self, 'vh_scheduler'):
            self.vh_scheduler.load_state_dict(training_state['vh_scheduler'])
        if 'vla_scheduler' in training_state and hasattr(self, 'vla_scheduler'):
            self.vla_scheduler.load_state_dict(training_state['vla_scheduler'])

class OpenVLAPPO:
    def __init__(self, all_args, policy: OpenVLAPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = getattr(self.args, "ppo_clip", 0.2)
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn
        self.rank = self.policy.device.index

        # 系数调度器：直接从args中检测各系数类型并初始化
        self._init_coef_schedule()

    def _init_coef_schedule(self):
        """初始化系数调度器：检测args中各系数的类型并构建统一的调度函数"""
        # 从args中获取各系数的定义（可能是常数、分段线性或其他调度类型）
        coef_defs = {
            'policy': parse_schedule_def(getattr(self.args, 'policy_coef', 1.0)),  # 支持字符串/列表/常数
            'value': parse_schedule_def(getattr(self.args, 'value_coef', 1.0)),
            'entropy': parse_schedule_def(getattr(self.args, 'alg_entropy_coef', 0.0)),
            'dist': parse_schedule_def(getattr(self.args, 'distribution_loss_coef', 0.0)),
        }

        # 构建统一的调度函数：progress -> dict
        def _unified_schedule(progress: float) -> dict:
            result = {}
            for name, definition in coef_defs.items():
                # 使用coef_schedules.py中的eval_schedule统一处理各种格式
                value = eval_schedule(definition, progress)
                result[name] = float(value if value is not None else 0.0)
            return result

        self.coef_schedule = _unified_schedule

    def train_ppo_step(self, idx, total, batch, progress=None):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)  # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)

        # 动态损失权重（基于训练进度 progress ∈ [0,1]）
        if progress is None:
            progress = 0.0
        # 直接获取当前进度对应的所有系数值
        coefs = self.coef_schedule(progress)

        # Policy loss
        logprob, entropy, values, logits = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        # surr1 是标准的策略梯度项
        surr1 = ratio * advantages
        # surr2 是裁剪后的策略梯度项
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        # minimize
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Value loss
        # value_preds 来自 batch，是上一轮旧网络的输出，不应再走梯度 否则 value_pred_clipped 里会有一条从旧 value_preds 过来的梯度路径，
        # 虽最终不会更新旧网络，但浪费显存，且容易在复杂模型里引入数值不稳定
        value_preds = value_preds.detach()
        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()

        value_loss = value_loss.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Distribution-shape loss（对每个动作自由度的分布进行“高斯形状”约束，bins=256）
        # 解释：action_len = actions.shape[1]，通常对应6个自由度：(x, y, z, roll, yaw, pitch)
        dist_coef = float(coefs.get("dist", 0.0))
        sigma_record = None
        if dist_coef != 0.0:
            logits_prepared = logits
            total_action_dims = logits_prepared.size(1)
            action_dim = getattr(self.args, "dist_action_dim", None) or total_action_dims
            if action_dim <= 0:
                action_dim = total_action_dims
            if total_action_dims % action_dim != 0:
                action_dim = total_action_dims

            constrained_dims = getattr(self.args, "dist_constrained_dims", action_dim)
            constrained_dims = max(0, min(int(constrained_dims), action_dim))

            max_chunk_num = getattr(self.args, "dist_max_chunk_num", 1)
            if max_chunk_num is not None:
                max_chunk_num = max(1, int(max_chunk_num))

            dist_loss, sigma_record = compute_diss_loss(
                logits_prepared,
                action_dim=action_dim,
                constrained_dims=constrained_dims,
                max_chunk_num=max_chunk_num,
                loss_type=str(getattr(self.args, "dist_loss_type", "simplest")),
                tau=float(getattr(self.args, "dist_tau", 0.0)),
                trunc_k=getattr(self.args, "dist_trunc_k", None),
                topk=getattr(self.args, "dist_topk", None),
                kernel_sigma=float(getattr(self.args, "kernel_sigma", 1.0)),
                min_sigma=float(getattr(self.args, "dist_min_sigma", 1e-3)),
                fixed_sigma=getattr(self.args, "dist_fixed_sigma", None),
                peak_window_bins=getattr(self.args, "dist_peak_window_bins", None),
                eps=float(getattr(self.args, "dist_eps", 1e-6)),
            )
        else:
            dist_loss = torch.tensor(0.0, device=logprob.device, dtype=logprob.dtype)
            sigma_record = torch.zeros(1, device=logprob.device, dtype=logprob.dtype)

        # Total loss - 使用统一的系数调度结果
        loss = (coefs["policy"] * policy_loss
                + coefs["value"] * value_loss
                - coefs["entropy"] * entropy_loss
                + coefs["dist"] * dist_loss)
        loss /= self.args.alg_gradient_accum

        # 是否到达一次累积的“同步点”
        is_sync_step = (idx % self.args.alg_gradient_accum == self.args.alg_gradient_accum - 1) or (idx == total - 1)

        # 分布式下，中间步不做梯度同步
        # 检查是否为分布式训练且支持 no_sync
        import torch.distributed as dist
        is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # 修复：no_sync 应该作用在 DDP 包裹后的模型 self.policy.vla 上
        if is_distributed and not is_sync_step:
            # 使用 no_sync() 关闭中间步的梯度同步，仅在同步步进行 all-reduce
            with self.policy.vla.no_sync():
                loss.backward()
        else:
            loss.backward()

        if is_sync_step:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm)
            self.policy.vh_optimizer.step()
            self.policy.vla_optimizer.step()

            self.policy.vh_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            
            # 当前使用的损失权重（便于监控）
            policy_coef=coefs["policy"],
            value_coef=coefs["value"],
            entropy_coef=coefs["entropy"],

            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            value_clip_ratio=value_clip_ratio.item(),
            value_old_mean=value_preds.mean().item(),
            values_mean=values.mean().item(),
            returns_mean=returns.mean().item(),
            returns_norm_mean=returns_norm.mean().item(),
            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),

            distribution_loss=dist_loss.item(),
            distribution_loss_coef=coefs["dist"],
            distribution_sigma_mean=sigma_record.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_grpo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        # value_preds = torch.tensor(value_preds).to(**self.tpdv)
        # returns = torch.tensor(returns).to(**self.tpdv_vn) # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)

        # Policy loss
        logprob, entropy, values, _ = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla, self.ppo_grad_norm)
            self.policy.vla_optimizer.step()

            # 调用学习率调度器（GRPO 只更新 VLA 参数）
            if hasattr(self.policy, 'vla_scheduler') and self.policy.vla_scheduler is not None:
                self.policy.vla_scheduler.step()

            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_ppo(self, buffer, progress=None):
        train_info = defaultdict(lambda: [])
        train_steps = 0

        # buffer
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        # 跨 rank 对齐 mini-batch 数：取全局最大，保证 backward 次数一致
        import torch
        import torch.distributed as dist
        global_mbc = minibatch_count
        if dist.is_initialized():
            t = torch.tensor([minibatch_count], device=self.policy.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            global_mbc = int(t.item())

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator(target_mini_batches=global_mbc)

            train_iter = enumerate(data_generator)
            for idx, batch in tqdm(
                    train_iter,
                    total=global_mbc,
                    desc=f"train [rank {self.rank}]",
                    position=self.rank + 1,
                    leave=(self.rank == 0)):
                # 训练进度（0-1）：使用来自外层 episode 的全局进度, 若不存在则始终使用初始值
                info = self.train_ppo_step(idx, global_mbc, batch, progress=progress)
                for key, value in info.items():
                    train_info[key].append(value)
                train_steps += batch[-1].shape[0]

            # 调用学习率调度器 -- 同一个epoch内学习率一致，这是更常见的使用方式
            if hasattr(self.policy, 'vh_scheduler') and self.policy.vh_scheduler is not None:
                self.policy.vh_scheduler.step()
            if hasattr(self.policy, 'vla_scheduler') and self.policy.vla_scheduler is not None:
                self.policy.vla_scheduler.step()

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)
        final_info["train_steps"] = train_steps

        return final_info

    def train_grpo(self, buffer):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_grpo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(
                    enumerate(data_generator),
                    total=minibatch_count,
                    desc=f"train [rank {self.rank}]",
                    position=self.rank + 1,
                    leave=(self.rank == 0)):
                info = self.train_grpo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info
