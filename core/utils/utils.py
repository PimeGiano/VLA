import numpy as np
import torch


def parse_eval_seeds_arg(val):
    """将 CLI 输入的 eval_seeds 字符串解析为 list[int] 或 None。
    - None/空字符串/字符串 'none' -> None
    - 支持以空格或逗号分隔的数字，如 "0 1 2" 或 "0,1,2"
    - 若存在非整数 token，返回 None
    Source: core/utils/args.py 参数定义与此处解析保持一致
    """
    if val is None:
        return None
    # 有些场景旧脚本可能仍传 list[int]
    if isinstance(val, list):
        # 确保元素为 int，否则返回 None
        try:
            return [int(x) for x in val]
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return None
        if s.lower() == "none":
            return None
        import re
        parts = re.split(r"[\s,]+", s)
        out = []
        for p in parts:
            if not p:
                continue
            # 允许负数种子（尽管不常见）
            if p.lstrip("+-").isdigit():
                out.append(int(p))
            else:
                return None
        return out if len(out) > 0 else None
    # 其它类型一律按 None 处理
    return None


def to_numpy(x, dtype=None):
    """
    Convert input to numpy array with optional dtype.
    Supports torch.Tensor, list, and np.ndarray.
    """
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = x.cpu().numpy()
    elif isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError(f"Unsupported type for to_numpy: {type(x)}")
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    return x


def to_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list):
        return torch.tensor(x)
    else:
        raise TypeError(f"Unsupported type for to_tensor: {type(x)}")


def to_tensor_device(x, device):
    """
    将输入转换为位于指定 device 上的 torch.Tensor。
    支持以下输入类型：
    - torch.Tensor：直接迁移到目标 device
    - np.ndarray：使用 torch.from_numpy 转换后迁移到目标 device
    - list[torch.Tensor]：先在 dim=0 进行堆叠，再迁移到目标 device
    - list[np.ndarray]：先用 np.stack 在 axis=0 堆叠，再转换并迁移到目标 device

    参数:
    - x: 输入数据（tensor / ndarray / list[tensor] / list[ndarray]）
    - device: 目标设备（如 torch.device("cuda:0")）
    """
    if isinstance(x, torch.Tensor):
        # 直接迁移设备
        return x.to(device)
    elif isinstance(x, list):
        # 仅在存在元素时处理，空列表视为非法输入
        if len(x) == 0:
            raise ValueError("to_tensor_device: 不支持空列表作为输入")
        first = x[0]
        if isinstance(first, torch.Tensor):
            # list[tensor]，保持 batch 维为 0
            return torch.stack([v.to(device) for v in x], dim=0)
        elif isinstance(first, np.ndarray):
            # list[np.ndarray]，先用 numpy 堆叠再转 tensor
            return torch.from_numpy(np.stack(x, axis=0)).to(device)
        else:
            raise ValueError(f"to_tensor_device: 不支持的列表元素类型: {type(first)}")
    elif isinstance(x, np.ndarray):
        # 直接从 numpy 转 tensor 并迁移设备
        return torch.from_numpy(x).to(device)
    else:
        raise ValueError(f"to_tensor_device: 不支持的输入类型: {type(x)}")



def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()  # torch.tensor(3.0).tolist() --> {float} 3.0
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (int, float, np.integer, np.floating)):
        return x
    elif isinstance(x, list):
        return x
    elif isinstance(x, str):
        return x
    else:
        raise TypeError(f"Unsupported type for to_list: {type(x)}")


def to_mean(x):
    """处理输入值并返回其均值或原始值，兼容list中为str的情况"""
    if hasattr(x, 'to'):
        return float(x.to(torch.float32).mean().item())
    elif isinstance(x, np.ndarray):
        # numpy数组直接求均值
        return float(np.mean(x))
    elif isinstance(x, list):
        # list内部可能是str
        if all(isinstance(i, (int, float, np.integer, np.floating)) for i in x):
            return float(np.mean(x))
        else:
            return x  # 如果有str，直接返回原list
    elif isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    else:
        return x