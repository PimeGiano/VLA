import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def simplest_gaussian_shape_kl_loss(
    logits: torch.Tensor,
    action_len: Optional[int] = None,
    topk: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    step_weights: Optional[Sequence[float]] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    分布形状约束（基于 KL(p||q) 的可反向传播损失）。

    语义：
        - p 为 softmax(logits) 在离散 bins 上的分布（K=256）。
        - 先用 p 的一阶二阶矩估计均值 μ 与方差 σ^2（对齐位置与尺度），
          再在同一离散坐标上构造离散高斯 q，返回 KL(p||q)。

    输入：
        logits: [B, action_len, 256]
        action_len: 若提供，仅取前 action_len 个自由度
        topk: 若提供，仅在每个 (b,a) 的选中 bins 上计算（top-|z_logits|）
        mask: 可广播到 [B, action_len, 256] 的布尔掩码
        step_weights: 对每个动作自由度加权（长度需等于 action_len）

    返回：
        标量张量：KL(p||q) 的平均值
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be rank-3 like [B, action_len, 256], got {tuple(logits.shape)}")

    # 统一 float32 计算，提升数值稳定性
    x = logits.float()  # [B, L, K]
    B, L, K = x.shape

    L_eff = L if action_len is None else min(int(action_len), L)
    x = x[:, :L_eff, :]

    # p: softmax 概率分布（沿 bins 维）
    p = torch.softmax(x, dim=-1)  # [B, L_eff, K]
    log_p = torch.log(p.clamp_min(eps))

    # 离散坐标 t（对称映射到 [-1, 1]）
    t = torch.linspace(-1.0, 1.0, K, device=x.device, dtype=x.dtype)

    # 由 p 估计均值与方差（逐 (b,a)）
    mu = (p * t).sum(dim=-1, keepdim=True)                          # [B, L_eff, 1]
    var = (p * (t - mu) ** 2).sum(dim=-1, keepdim=True) + eps       # [B, L_eff, 1]
    std = var.sqrt()

    # 在相同坐标上构造离散高斯 q，并归一化
    s = (t - mu) / std                                              # [B, L_eff, K]
    q_raw = torch.exp(-0.5 * (s ** 2))                              # 高斯核
    q = q_raw / q_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_q = torch.log(q.clamp_min(eps))

    # 逐元素 KL 项：p * (log p - log q)
    kl_elem = p * (log_p - log_q)                                   # [B, L_eff, K]

    # 选择掩码：默认全选
    select = torch.ones_like(kl_elem, dtype=torch.bool)

    # 若指定 topk：沿 bins 维根据 p 选择概率质量最大的 top-k（更直观）
    if topk is not None and 0 < topk < K:
        _, idx = torch.topk(p, k=topk, dim=-1, largest=True, sorted=False)
        topk_mask = torch.zeros_like(select)
        topk_mask.scatter_(-1, idx, True)
        select = select & topk_mask

    # 外部 mask（可广播）；若给定且长度更长则裁剪到 L_eff
    if mask is not None:
        m = mask.to(dtype=torch.bool, device=select.device)
        if m.dim() == 3 and m.size(1) != L_eff:
            if m.size(1) >= L_eff:
                m = m[:, :L_eff, :]
            else:
                raise ValueError(
                    f"mask's middle dimension ({m.size(1)}) is shorter than effective action_len ({L_eff})"
                )
        select = select & m

    # 动作维权重（与 action_len 对齐）
    if step_weights is not None:
        w = torch.as_tensor(step_weights, dtype=kl_elem.dtype, device=kl_elem.device)
        if w.numel() != L_eff:
            raise ValueError(f"step_weights length ({w.numel()}) must equal action_len ({L_eff})")
        kl_elem = kl_elem * w.view(1, L_eff, 1)

    denom = select.float().sum().clamp_min(1.0)
    loss = (kl_elem * select.float()).sum() / denom
    sigma_record = std.squeeze(-1).detach()
    return loss, sigma_record


def gaussian_shape_kl_loss(
    logits: torch.Tensor,
    action_len: Optional[int] = None,
    trunc_k: float = 3.0,  # 3σ 截断窗口
    min_sigma: float = 1e-3,
    topk: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    step_weights: Optional[Sequence[float]] = None,
    eps: float = 1e-6,
    tau: float = 0.0,              # 温度参数 τ：τ<=0 使用 argmax 中心；τ>0 使用 soft-argmax
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    高斯形状约束（截断窗口内的 KL(p||q)）。

    设计要点：
    - p = softmax(logits)
    - 中心 μ：若 τ<=0 则使用 argmax(p)（硬中心，不可导）；若 τ>0 则使用 soft-argmax：μ = Σ t·softmax(logits/τ)
    - 目标 q 为以 μ 为中心的离散高斯；方差由 p 的二阶矩估计，sigma>=min_sigma
    - 仅在窗口 |t - mu| <= trunc_k * sigma 范围内进行重归一化后计算 KL
    - 若 trunc_k<=0 或为 inf，则关闭截断窗口（在全域上计算 KL）
    - 可选：与 topk / 外部 mask 共同收缩窗口
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be rank-3 like [B, action_len, 256], got {tuple(logits.shape)}")

    x = logits.float()
    B, L, K = x.shape
    L_eff = L if action_len is None else min(int(action_len), L)
    x = x[:, :L_eff, :]

    p = torch.softmax(x, dim=-1)  # [B, L_eff, K]

    # 离散坐标：[-1, 1]，展开到 [B, L_eff, K] 以便与 gather 对齐
    t = torch.linspace(-1.0, 1.0, K, device=x.device, dtype=x.dtype).view(1, 1, K).expand(B, L_eff, K)

    old_version = False
    if old_version:
        # 中心 μ：τ<=0 使用 argmax（不可导），τ>0 使用 soft-argmax（可导）
        if float(tau) <= 0.0:
            k_star = p.argmax(dim=-1, keepdim=True)                     # [B, L_eff, 1], long
            mu = t.gather(-1, k_star)                                   # [B, L_eff, 1]
        else:
            tau_eff = float(max(tau, eps))                              # 保障数值稳定
            p_center = torch.softmax(x / tau_eff, dim=-1)               # [B, L_eff, K]
            mu = (p_center * t).sum(dim=-1, keepdim=True)               # [B, L_eff, 1]

        # 由 p 估计 sigma（围绕 mu 的二阶矩），并设置下限
        diff = t - mu                             # [B, L_eff, K]
        var = (p * (diff ** 2)).sum(dim=-1, keepdim=True)
        var = torch.clamp(var, min=min_sigma * min_sigma)
        sigma = var.sqrt()

        # 目标离散高斯 q（未归一化）
        s = diff / sigma
        q_raw = torch.exp(-0.5 * (s ** 2))

    else:
        if float(tau) <= 0.0:
            k_star = p.argmax(dim=-1, keepdim=True)
            peak_coord = t.gather(-1, k_star)
            mu = peak_coord
        else:
            tau_eff = float(max(tau, eps))
            p_center = torch.softmax(x / tau_eff, dim=-1)
            mu_soft = (p_center * t).sum(dim=-1, keepdim=True)
            k_star = p.argmax(dim=-1, keepdim=True)
            peak_coord = t.gather(-1, k_star)
            mu = peak_coord + (mu_soft - mu_soft.detach())

        diff = t - mu

        # 目标峰值概率
        target_peak = p.gather(-1, k_star).squeeze(-1).clamp_min(eps)

        # 预先计算与峰值的平方差
        diff_peak = t - peak_coord
        diff_peak_sq = diff_peak.square()

        def q_peak_at_sigma(sigma: torch.Tensor) -> torch.Tensor:
            sigma_sq = torch.clamp(sigma ** 2, min=min_sigma * min_sigma)
            exponent = torch.exp(-0.5 * diff_peak_sq / sigma_sq.unsqueeze(-1))
            sum_exp = exponent.sum(dim=-1).clamp_min(eps)
            peak_val = exponent.gather(-1, k_star).squeeze(-1)
            return peak_val / sum_exp

        sigma_low = torch.full_like(target_peak, min_sigma)
        sigma_high = torch.full_like(target_peak, 1.0)

        q_high = q_peak_at_sigma(sigma_high)
        for _ in range(10):
            mask = q_high > target_peak
            if not mask.any():
                break
            sigma_high = torch.where(mask, sigma_high * 2.0, sigma_high)
            q_high = torch.where(mask, q_peak_at_sigma(sigma_high), q_high)

        sigma = sigma_high.clone()

        almost_one = target_peak >= (1.0 - 1e-6)
        # sigma = torch.where(almost_one, torch.full_like(sigma, min_sigma), sigma)
        sigma_low = torch.where(almost_one, torch.full_like(sigma_low, min_sigma), sigma_low)
        sigma_high = torch.where(almost_one, torch.full_like(sigma_high, min_sigma), sigma_high)

        for _ in range(25):
            sigma_mid = 0.5 * (sigma_low + sigma_high)
            q_mid = q_peak_at_sigma(sigma_mid)
            mask = q_mid > target_peak
            sigma_low = torch.where(mask, sigma_mid, sigma_low)
            sigma_high = torch.where(mask, sigma_high, sigma_mid)

        sigma = sigma_high.unsqueeze(-1)

        q = torch.exp(-0.5 * (diff / sigma) ** 2)
        q_raw = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

    # 截断窗口：|t - mu| <= trunc_k * sigma；当 trunc_k<=0 或非有限（如 inf）时，不进行截断
    if math.isfinite(float(trunc_k)) and trunc_k > 0:
        window = (diff.abs() <= (trunc_k * sigma))
    else:
        window = torch.ones_like(diff, dtype=torch.bool)

    # 与 topk / 外部 mask 合并窗口
    combined = window
    if topk is not None and 0 < topk < K:
        _, idx = torch.topk(p, k=topk, dim=-1, largest=True, sorted=False)
        topk_mask = torch.zeros_like(combined)
        topk_mask.scatter_(-1, idx, True)
        combined = combined & topk_mask

    if mask is not None:
        m = mask.to(dtype=torch.bool, device=x.device)
        if m.dim() == 3 and m.size(1) != L_eff:
            if m.size(1) >= L_eff:
                m = m[:, :L_eff, :]
            else:
                raise ValueError(
                    f"mask's middle dimension ({m.size(1)}) is shorter than effective action_len ({L_eff})"
                )
        combined = combined & m

    # 在组合窗口内重归一化 p 和 q
    comb_f = combined.float()
    p_sum = (p * comb_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    q_sum = (q_raw * comb_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    p_w = (p * comb_f) / p_sum
    q_w = (q_raw * comb_f) / q_sum

    log_p_w = torch.log(p_w.clamp_min(eps))
    log_q_w = torch.log(q_w.clamp_min(eps))

    # 对每个 (b, a) 计算 KL，再做均值
    # KL(p‖q)：约束模型分布 p 要“贴近”目标分布 q; KL(q‖p)：约束目标分布覆盖的区域 p 不能忽略;
    kl_pq_per = (p_w * (log_p_w - log_q_w)).sum(dim=-1)  # [B, L_eff]

    # 使用 F.kl_div 计算逐元素 KL：期望 target 为概率、input 为 log 概率
    # F.kl_div(log_q_w, p_w, reduction="none").sum(dim=-1) 效果一样

    if step_weights is not None:
        w = torch.as_tensor(step_weights, dtype=kl_pq_per.dtype, device=kl_pq_per.device)
        if w.numel() != L_eff:
            raise ValueError(f"step_weights length ({w.numel()}) must equal action_len ({L_eff})")
        kl_pq_per = kl_pq_per * w.view(1, L_eff)

    sigma_record = sigma.squeeze(-1).detach()
    return kl_pq_per.mean(), sigma_record


def gaussian_kernel_smooth_kl_loss(
    logits: torch.Tensor,
    action_len: Optional[int] = None,
    kernel_sigma: float = 0.1,
    trunc_k: Optional[float] = None,
    topk: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    step_weights: Optional[Sequence[float]] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于“对模型分布 p 进行高斯核平滑得到 q”，再计算 KL 的分布正则。

    语义：
        - p = softmax(logits) ∈ R^{B×L×K}
        - 在离散坐标 t_k∈[-1,1] 上构造 K×K 的高斯核矩阵 Kmat[k,j] = exp(-0.5 * ((t_k - t_j)/sigma)^2)
        - 对 Kmat 做【按列归一化】（每列和为 1），然后 q = p @ Kmat^T
        - 可选：对核做局部截断（|t_k - t_j| ≤ trunc_k * sigma），以及对 KL 的 bins 维应用 topk/外部 mask，并在窗口内对 p、q 重归一化
        - 最终计算 KL(p||q) 或 KL(q||p)，并按 step_weights 加权后取均值

    参数：
        logits: [B, action_len, K]
        action_len: 若提供，仅取前 action_len 个自由度
        kernel_sigma: 高斯核带宽 sigma（标量），必须 > 0
        trunc_k: 可选截断窗口半径（单位：sigma），如 3.0 表示 |t_k - t_j| ≤ 3*sigma；None/<=0/非有限 则关闭
        topk: 可选 KL 计算仅在每个 (b,l) 的 top-k bins 上进行
        mask: 可广播到 [B, L_eff, K] 的布尔掩码
        step_weights: 长度为 L_eff 的权重

        eps: 数值稳定项

    返回：
        标量张量：最终 KL 的 batch/步 平均
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be rank-3 like [B, action_len, K], got {tuple(logits.shape)}")

    if float(kernel_sigma) <= 0:
        raise ValueError(f"kernel_sigma must be > 0, got {kernel_sigma}")

    # 统一 float32 计算，提升数值稳定性
    x = logits.float()  # [B, L, K]
    B, L, K = x.shape

    # 对齐 action_len
    L_eff = L if action_len is None else min(int(action_len), L)
    x = x[:, :L_eff, :]

    # p: softmax 概率分布（沿 bins 维）
    p = torch.softmax(x, dim=-1)  # [B, L_eff, K]

    # 离散坐标 t ∈ [-1, 1]
    t = torch.linspace(-1.0, 1.0, K, device=x.device, dtype=x.dtype)
    t_expanded = t.view(1, 1, K)

    # 记录当前分布的标准差，供外部监控
    mu = (p * t_expanded).sum(dim=-1, keepdim=True)
    var = (p * (t_expanded - mu) ** 2).sum(dim=-1, keepdim=True) + eps
    std = var.sqrt()

    # 构造 K×K 的 pairwise 距离矩阵 Δ[k,j] = t_k - t_j
    # 说明：按列归一化（列和=1），确保 q = p @ Knorm^T 仍为概率分布
    diff = t.view(K, 1) - t.view(1, K)                 # [K, K]

    # 高斯核权重矩阵（可选局部截断）
    sigma = torch.as_tensor(float(kernel_sigma), dtype=x.dtype, device=x.device)
    Kmat = torch.exp(-0.5 * (diff * diff) / (sigma * sigma))  # [K, K]

    # 截断窗口：|Δ| ≤ trunc_k * sigma
    if (trunc_k is not None) and math.isfinite(float(trunc_k)) and (float(trunc_k) > 0.0):
        window = (diff.abs() <= (float(trunc_k) * sigma))      # [K, K]
        Kmat = Kmat * window.to(dtype=Kmat.dtype)

    # 按列归一化，确保列和为 1（从而 q 仍归一化）
    colsum = Kmat.sum(dim=0, keepdim=True).clamp_min(eps)
    Knorm = Kmat / colsum

    # 计算 q = p @ Knorm^T
    p_flat = p.reshape(-1, K)                                 # [B*L_eff, K]
    q_flat = torch.matmul(p_flat, Knorm.t())                  # [B*L_eff, K]
    q = q_flat.view(B, L_eff, K)

    # ----------------------
    # KL 计算前的 bins 选择窗口（topk / 外部 mask），并对 p、q 重归一化
    # ----------------------
    select = torch.ones_like(q, dtype=torch.bool)             # [B, L_eff, K]

    if topk is not None and 0 < topk < K:
        _, idx = torch.topk(p, k=topk, dim=-1, largest=True, sorted=False)
        topk_mask = torch.zeros_like(select)
        topk_mask.scatter_(-1, idx, True)
        select = select & topk_mask

    if mask is not None:
        m = mask.to(dtype=torch.bool, device=select.device)
        if m.dim() == 3 and m.size(1) != L_eff:
            if m.size(1) >= L_eff:
                m = m[:, :L_eff, :]
            else:
                raise ValueError(
                    f"mask's middle dimension ({m.size(1)}) is shorter than effective action_len ({L_eff})"
                )
        select = select & m

    comb_f = select.float()
    p_sel = (p * comb_f)
    q_sel = (q * comb_f)
    p_sum = p_sel.sum(dim=-1, keepdim=True).clamp_min(eps)
    q_sum = q_sel.sum(dim=-1, keepdim=True).clamp_min(eps)
    p_w = p_sel / p_sum
    q_w = q_sel / q_sum

    log_p_w = torch.log(p_w.clamp_min(eps))
    log_q_w = torch.log(q_w.clamp_min(eps))

    # 逐 (b,l) 的 KL 值：KL(p‖q)
    kl_pq_per = (p_w * (log_p_w - log_q_w)).sum(dim=-1)      # [B, L_eff]

    # 动作维权重
    if step_weights is not None:
        w = torch.as_tensor(step_weights, dtype=kl_pq_per.dtype, device=kl_pq_per.device)
        if w.numel() != L_eff:
            raise ValueError(f"step_weights length ({w.numel()}) must equal action_len ({L_eff})")
        kl_pq_per = kl_pq_per * w.view(1, L_eff)

    sigma_record = std.squeeze(-1).detach()
    return kl_pq_per.mean(), sigma_record

__all__ = [
    "gaussian_shape_kl_loss",
    "simplest_gaussian_shape_kl_loss",
    "gaussian_kernel_smooth_kl_loss",
]
