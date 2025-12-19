#-*-coding:utf-8-*-
from __future__ import annotations

import math
from typing import Optional

import torch


def compute_diss_loss(
    logits: torch.Tensor,
    *,
    action_dim: int = 7,
    constrained_dims: int = 6,
    max_chunk_num: Optional[int] = None,
    loss_type: str = "simplest",
    tau: float = 0.0,
    trunc_k: Optional[float] = None,
    topk: Optional[int] = None,
    peak_window_bins: Optional[int] = None,
    kernel_sigma: float = 1.0,
    min_sigma: float = 1e-3,
    fixed_sigma: None,
    eps: float = 1e-6,
):
    """
    对动作的前若干维度施加分布正则约束，并提供多种形状约束类型。

    Args:
        logits: 原始 logits，默认是 [B, action_dim * num_chunks, action_bins]
            也支持  [B, action_bins, action_dim * num_chunks]。
        action_dim: 每个动作 chunk 的自由度（默认 7）。
        constrained_dims: 每个动作中需要约束的前几个维度（默认 6）。
        max_chunk_num: 仅对前 max_chunk_num 个 chunk 计算损失；None 表示全部 chunk。
        loss_type: 约束类型，可选 ``simplest``、``gaussian_shape``、``gaussian_kernel``。
        tau: ``gaussian_shape`` 中用来决定中心的温度（τ<=0 使用 argmax）。
        trunc_k: 截断窗口半径（单位：σ）；None/非正/非有限值表示不截断。
        topk: 若给定，仅保留概率最大的 top-k bins 参与 KL。
        peak_window_bins: 仅在 ``loss_type`` 为 ``gaussian_kernel`` 时生效，
            指定以概率峰值为中心、仅在 ±window bins 内计算 KL。
        kernel_sigma: ``gaussian_kernel`` 的高斯核带宽。
        min_sigma: ``gaussian_shape`` 中 σ 的下限。
        fixed_sigma:  固定sigma。
        eps: 数值稳定常数。
    """
    selected_logits = _select_action_logits(
        logits,
        action_dim=action_dim,
        constrained_dims=constrained_dims,
        max_chunk_num=max_chunk_num,
    )  # [B, num_chunks, max_dims, action_bins]

    if selected_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    selected_logits = selected_logits.float()
    loss_type = loss_type.lower().strip()

    if loss_type in {"simplest", "gaussian", "gaussian_simple"}:
        loss, sigma_record = _simplest_gaussian_shape_kl_loss(
            selected_logits,
            tau=tau,
            trunc_k=trunc_k,
            topk=topk,
            min_sigma=min_sigma,
            fixed_sigma=fixed_sigma,
            eps=eps,
        )
    elif loss_type in {"gaussian_shape", "shape"}:
        loss, sigma_record = _gaussian_shape_kl_loss(
            selected_logits,
            tau=tau,
            trunc_k=trunc_k,
            topk=topk,
            min_sigma=min_sigma,
            eps=eps,
        )
    elif loss_type in {"gaussian_kernel", "kernel"}:
        loss, sigma_record = _gaussian_kernel_smooth_kl_loss(
            selected_logits,
            kernel_sigma=kernel_sigma,
            trunc_k=trunc_k,
            topk=topk,
            peak_window_bins=peak_window_bins,
            eps=eps,
        )
    else:
        raise ValueError(
            f"Unsupported dist loss type '{loss_type}'. "
            "Expected one of: 'simplest', 'gaussian_shape', 'gaussian_kernel'."
        )

    return loss.to(dtype=logits.dtype), sigma_record


def _select_action_logits(
    logits: torch.Tensor,
    *,
    action_dim: int,
    constrained_dims: int,
    max_chunk_num: Optional[int],
) -> torch.Tensor:
    if logits is None:
        raise ValueError("logits must be provided to compute_diss_loss.")

    if logits.dim() != 3:
        raise ValueError(
            "logits must have shape [batch, action_dim * num_chunks, action_bins] "
            f"or [batch, action_bins, action_dim * num_chunks]; got {tuple(logits.shape)}"
        )

    dim1, dim2 = logits.shape[1], logits.shape[2]

    if dim2 >= dim1:
        # [B, action_dim*num_chunks, bins]
        total_action_dims = dim1
        action_bins = dim2
        logits_prepared = logits
    else:
        # [B, bins, action_dim*num_chunks] -> [B, action_dim*num_chunks, bins]
        total_action_dims = dim2
        action_bins = dim1
        logits_prepared = logits.transpose(1, 2).contiguous()

    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if total_action_dims % action_dim != 0:
        raise ValueError(
            f"total action dims ({total_action_dims}) not divisible by action_dim ({action_dim})"
        )

    max_dims = max(min(constrained_dims, action_dim), 0)
    if max_dims == 0:
        return torch.empty(0, device=logits.device, dtype=logits.dtype)

    num_chunks = total_action_dims // action_dim

    # 先 reshape 回 [B, num_chunks, action_dim, action_bins]，再选取每个 chunk 的前 max_dims 个维度
    logits_view = logits_prepared.view(
        logits_prepared.size(0), num_chunks, action_dim, action_bins
    )

    if max_chunk_num is None:
        chunk_limit = num_chunks
    else:
        chunk_limit = max(int(max_chunk_num), 0)
    chunk_limit = min(chunk_limit, num_chunks)

    if max_dims == 0 or chunk_limit == 0:
        return torch.empty(0, device=logits.device, dtype=logits.dtype)

    selected = logits_view[:, :chunk_limit, :max_dims, :].reshape(
        logits_view.size(0), chunk_limit * max_dims, action_bins
    )

    return selected


def _simplest_gaussian_shape_kl_loss(
    logits: torch.Tensor,
    *,
    tau: float = -1.0,
    trunc_k: Optional[float],
    topk: Optional[int],
    min_sigma: float,
    fixed_sigma: Optional[float] = None,
    eps: float,
) -> torch.Tensor:
    """
    最简单的高斯形状 KL 损失，均值使用原分布的期望，标准差使用原分布的标准差。
    可选地配合截断 / top-k 仅在局部区域计算。

    Args:
        logits: [B, L, K] 形式的 logits。
        trunc_k: σ 的倍数，限制 |t - μ| ≤ trunc_k * σ 的 bins。
        topk: 仅保留概率最大的 top-k bins。
        eps: 数值稳定常数。
    """
    tau_eff = float(max(abs(tau), eps))
    p = torch.softmax(logits/tau_eff, dim=-1)
    t = _bin_coordinates(logits).expand_as(p)

    # 中心 μ：τ<=0 使用 argmax；τ>0 使用 soft-argmax，但保持峰值对应的离散位置
    if float(tau) < 0.0:
        k_star = p.argmax(dim=-1, keepdim=True)
        peak_coord = t.gather(-1, k_star)
        mu = peak_coord
    else:
        mu = (p * t).sum(dim=-1, keepdim=True)

    diff = t - mu
    if fixed_sigma is not None:
        std = torch.as_tensor(
            float(fixed_sigma), dtype=logits.dtype, device=logits.device
        ).unsqueeze(-1)

        # just for record
        mu_std = (p * t).sum(dim=-1, keepdim=True)
        diff_std = t - mu_std
        var = (p * diff_std.square()).sum(dim=-1, keepdim=True).clamp_min(eps)
        sigma_record = torch.sqrt(var)
    else:
        mu_std = (p * t).sum(dim=-1, keepdim=True)
        diff_std = t - mu_std
        var = (p * diff_std.square()).sum(dim=-1, keepdim=True).clamp_min(eps)
        std = torch.sqrt(var)
        sigma_record = std.clone()
        std_min = torch.as_tensor(
            min_sigma, dtype=std.dtype, device=std.device
        )
        std = torch.maximum(std, std_min)

    q = torch.exp(-0.5 * (diff / std) ** 2)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

    select = _build_selection_mask(
        p,
        diff,
        std,
        trunc_k=trunc_k,
        topk=topk,
    )
    kl = _masked_kl(p, q, select, eps=eps)
    return kl, sigma_record


def _gaussian_shape_kl_loss(
    logits: torch.Tensor,
    *,
    tau: float = 0.0,
    trunc_k: Optional[float],
    topk: Optional[int],
    min_sigma: float,
    eps: float,
) -> torch.Tensor:
    """
    高斯形状约束（截断窗口内的 KL(p||q)）。

    设计要点：
    - p = softmax(logits)
    - 中心 μ：若 τ<=0 则使用 argmax(p)（硬中心，不可导）；若 τ>0 则使用 soft-argmax：μ = Σ t·softmax(logits/τ)
    - 目标 q 的峰值与原分布峰值对齐；方差通过相对峰值的二阶矩估计，sigma>=min_sigma
    - 仅在窗口 |t - mu| <= trunc_k * sigma 范围内进行重归一化后计算 KL
    - 若 trunc_k<=0 或为 inf，则关闭截断窗口（在全域上计算 KL）
    - 可选：与 topk / 外部 mask 共同收缩窗口
    """
    tau_eff = float(max(abs(tau), eps))
    p = torch.softmax(logits / tau_eff, dim=-1)
    t = _bin_coordinates(logits).expand_as(p)

    # 中心 μ：τ<=0 使用 argmax；τ>0 使用 soft-argmax，但保持峰值对应的离散位置
    if float(tau) < 0.0:
        k_star = p.argmax(dim=-1, keepdim=True)
        peak_coord = t.gather(-1, k_star)
        mu = peak_coord
    else:
        mu_soft = (p * t).sum(dim=-1, keepdim=True)
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
    # q_low = q_peak_at_sigma(sigma_low)

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

    sigma_record = sigma_high.clone()
    sigma = torch.clamp(sigma_high, min=min_sigma).unsqueeze(-1)

    q = torch.exp(-0.5 * (diff / sigma) ** 2)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

    select = _build_selection_mask(
        p,
        diff,
        sigma,
        trunc_k=trunc_k,
        topk=topk,
    )
    kl = _masked_kl(p, q, select, eps=eps)
    return kl.mean(), sigma_record


def _gaussian_kernel_smooth_kl_loss(
    logits: torch.Tensor,
    *,
    kernel_sigma: float,
    trunc_k: Optional[float],
    topk: Optional[int],
    peak_window_bins: Optional[int],
    eps: float,
) -> torch.Tensor:
    """
    计算基于“高斯核平滑”的分布正则损失：
        先将预测分布 p = softmax(logits)，
        再以高斯核矩阵对其进行平滑得到 q，
        最后计算 KL(p || q)。

    这种损失鼓励模型输出的概率分布在离散坐标上更平滑，避免出现过度尖锐或抖动的预测。

    Args:
        logits: [B, L, K]  模型原始 logits
            - B: batch size
            - L: 动作长度/时间步（action_len）
            - K: 离散 bins 数（vocab 或离散化行动空间大小）
        kernel_sigma: 高斯核带宽 σ（平滑程度，越大越平滑）
        trunc_k: 可选核的截断窗口半径（单位为 σ）
                 若为 None 或 <=0，则不截断核
        topk: 可选，仅在概率较高的 top-k bins 上计算 KL
        peak_window_bins: 可选，仅保留距离概率峰值若干 bins 的局部窗口（-window~+window）参与 KL
        eps: 数值稳定常数，避免除零与 log(0)

    Returns:
        标量张量：所有样本与步平均后的 KL 散度
    """
    if float(kernel_sigma) <= 0.0:
        raise ValueError(f"kernel_sigma must be > 0, got {kernel_sigma}")

    p = torch.softmax(logits, dim=-1)
    t = _bin_coordinates(logits)
    bins = logits.size(-1)

    grid = t.view(bins)
    diff = grid.view(bins, 1) - grid.view(1, bins)

    # 构建高斯核矩阵
    sigma = torch.as_tensor(
        float(kernel_sigma), dtype=logits.dtype, device=logits.device
    )
    kernel = torch.exp(-0.5 * (diff / sigma) ** 2)

    # 可选截断：|t_k - t_j| > trunc_k * σ 时置零 “裁掉”过远的部分（超出一定倍数 σ 的邻域范围）
    trunc_val = None  # 用均值/方差刻画的截断窗口
    if trunc_k is not None:
        trunc_val = float(trunc_k)
    if trunc_val is not None and math.isfinite(trunc_val) and trunc_val > 0.0:
        window_kernel = diff.abs() <= (trunc_val * sigma)
        kernel = kernel * window_kernel.to(dtype=kernel.dtype)

    # 按列归一化核矩阵
    colsum = kernel.sum(dim=0, keepdim=True).clamp_min(eps)
    kernel = kernel / colsum

    # 对预测分布 p 应用核平滑产生 q
    p_flat = p.reshape(-1, bins)
    q_flat = torch.matmul(p_flat, kernel.t())
    q = q_flat.view_as(p)

    # just record sigma
    mu = (p * t).sum(dim=-1, keepdim=True)
    local_diff = t - mu
    var = (p * local_diff.square()).sum(dim=-1, keepdim=True).clamp_min(eps)
    std = torch.sqrt(var)
    std_min = torch.as_tensor(
        math.sqrt(eps), dtype=std.dtype, device=std.device
    )
    std = torch.maximum(std, std_min)

    select = _build_selection_mask(
        p,
        local_diff,
        std,
        trunc_k=trunc_k,
        topk=topk,
        peak_window_bins=peak_window_bins,
    )
    kl = _masked_kl(p, q, select, eps=eps)
    sigma_record = std.squeeze(-1).detach()
    return kl.mean(), sigma_record


def _build_selection_mask(
    p: torch.Tensor,
    diff: torch.Tensor,
    scale: torch.Tensor,
    *,
    trunc_k: Optional[float],
    topk: Optional[int],
    peak_window_bins: Optional[int] = None,
) -> torch.Tensor:
    """
    构建选择掩码，仅在指定窗口 / top-k / 峰值邻域内选择元素。
    """
    select = torch.ones_like(p, dtype=torch.bool)

    trunc_val = None
    if trunc_k is not None:
        trunc_val = float(trunc_k)
    if trunc_val is not None and math.isfinite(trunc_val) and trunc_val > 0.0:
        select = select & (diff.abs() <= (trunc_val * scale))

    if topk is not None:
        k = int(topk)
        bins = p.size(-1)
        if 0 < k < bins:
            _, idx = torch.topk(p, k=k, dim=-1, largest=True, sorted=False)
            topk_mask = torch.zeros_like(p, dtype=torch.bool)
            topk_mask.scatter_(-1, idx, True)
            select = select & topk_mask

    if peak_window_bins is not None:
        window = int(max(peak_window_bins, 0))
        bins = p.size(-1)
        if window < bins:
            # 仅保留距概率峰值（argmax）±window 范围内的 bins
            indices = torch.arange(bins, device=p.device).view(1, 1, -1)
            peak_idx = p.argmax(dim=-1, keepdim=True)
            peak_mask = (indices - peak_idx).abs() <= window
            select = select & peak_mask

    valid = select.sum(dim=-1, keepdim=True) == 0
    if valid.any():
        select = torch.where(valid, torch.ones_like(select), select)

    return select


def _masked_kl(
    p: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """
    计算在掩码(mask)范围内的 KL 散度（Kullback–Leibler Divergence）。

    用于当部分分布位置无效（例如被截断、padding、或仅子区间有效）时，
    对有效区域内的概率分布 p、q 计算归一化后的 KL(p || q)。

    Args:
        p: 源概率分布张量 [..., bins]，例如模型输出的 softmax 概率。
        q: 目标概率分布张量 [..., bins]，例如高斯模板分布。
        mask: 同形状布尔掩码，True/1 表示有效位置，False/0 表示忽略位置。
        eps: 数值稳定常数，用于防止除零与 log(0)。

    Returns:
        KL 散度张量 [...], 已按最后一维聚合求和，仅在 mask 区间内计算。
    """
    mask_f = mask.float()
    p_sel = p * mask_f
    q_sel = q * mask_f

    # 计算掩码范围内的总和（即子分布的归一化常数）。
    # clamp_min 防止全零情形下出现 divide-by-zero。
    p_sum = p_sel.sum(dim=-1, keepdim=True).clamp_min(eps)
    q_sum = q_sel.sum(dim=-1, keepdim=True).clamp_min(eps)

    # 在 mask 后重新归一化，使得有效区域内概率和=1。
    p_norm = p_sel / p_sum
    q_norm = q_sel / q_sum

    # 取对数（避免 log(0)）
    log_p = torch.log(p_norm.clamp_min(eps))
    log_q = torch.log(q_norm.clamp_min(eps))

    # 计算 KL 散度：KL(p‖q) = Σ p * (log p − log q)：约束模型分布 p 要“贴近”目标分布 q;
    kl = (p_norm * (log_p - log_q)).sum(dim=-1)
    return kl


def _bin_coordinates(logits: torch.Tensor) -> torch.Tensor:
    bins = logits.size(-1)
    coords = torch.linspace(
        -1.0, 1.0, bins, device=logits.device, dtype=logits.dtype
    )
    return coords.view(1, 1, bins)
