"""
utils.coef_schedules

- 统一的调度工具：线性插值、评估、损失系数解析
- 学习率调度（warmup 等）创建逻辑下沉到此，减少上层 openvla_train.py 体积

注：不改变现有训练流程与默认行为，仅做结构化与复用。
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import ast


# HF 学习率调度器（保持与原实现一致）
from transformers import get_constant_schedule_with_warmup

Number = Union[int, float]
Point = Tuple[Number, Number]
ScheduleDef = Union[
    Callable[[float], Number],  # 可调用：f(progress)->value
    Sequence[Point],            # 分段线性插值：[(t0, v0), (t1, v1), ...]
    Number,                     # 常数
]


class LinearInterpSchedule:
    """分段线性插值调度器
    - points: [(t0, v0), (t1, v1), ...]
      t 可为非降或非升（内部按 t 排序）；v 可上升/下降/或分段变化，均做线性插值
    - 区间外按端点值截断

    示例（线性下降且分段斜率变化）：
        LinearInterpSchedule([(0.0, 1.0), (0.5, 0.5), (1.0, 0.1)])
        效果：
          progress=0.0  -> 1.0 (list[0])
          progress=0.25 -> 0.75
          progress=0.5  -> 0.5 (list[1])
          progress=0.75 -> 0.3
          progress=1.0  -> 0.1 (list[2])
    """
    def __init__(self, points: Sequence[Point]):
        # points: [(t0, v0), (t1, v1), ...]
        self.points: List[Point] = sorted(points, key=lambda x: x[0])

    def __call__(self, progress: float) -> float:
        pts = self.points
        if not pts:
            return 0.0
        if progress <= pts[0][0]:
            return float(pts[0][1])
        if progress >= pts[-1][0]:
            return float(pts[-1][1])
        for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
            if t0 <= progress <= t1:
                if t1 == t0:
                    return float(v1)
                alpha = (progress - t0) / (t1 - t0)
                return float(v0 * (1 - alpha) + v1 * alpha)
        # fallback（理论上不会走到）
        return float(pts[-1][1])


def eval_schedule(sched: Optional[ScheduleDef], progress: float) -> Optional[float]:
    """评估单个调度定义；支持：
    - 可调用：f(progress)->value
    - 分段线性插值：[(t, v), ...]
    - 常数：number
    - None：返回 None
    """
    if sched is None:
        return None
    if callable(sched):
        return float(sched(progress))
    if isinstance(sched, (list, tuple)) and len(sched) > 0:
        return float(LinearInterpSchedule(sched)(progress))
    if isinstance(sched, (int, float)):
        return float(sched)
    return None


def resolve_coefs(
    progress: float,
    base: Dict[str, float],
    schedule: Optional[Union[Callable[[float, Dict[str, float]], Dict[str, float]], Dict[str, ScheduleDef]]],
) -> Dict[str, float]:
    """按照给定 schedule 解析当前步的损失权重。
    - base: 基础权重字典（会被就地更新并返回）
    - schedule:
        * 可调用：f(progress, base)->dict，与现有逻辑完全兼容
        * dict：{name: schedule_def}，其中 schedule_def 支持可调用/线性插值/常数
    """
    if schedule is None:
        return base

    # 可调用：直接让外部完整控制
    if callable(schedule):  # type: ignore[call-arg]
        upd = schedule(progress, base.copy())  # type: ignore[misc]
        if isinstance(upd, dict):
            for k, v in upd.items():
                if k in base and v is not None:
                    base[k] = float(v)
        return base

    # dict：逐项评估
    if isinstance(schedule, dict):
        for k, v in schedule.items():
            if k in base:
                ev = eval_schedule(v, progress)
                if ev is not None:
                    base[k] = float(ev)
        return base


# ==========================
# CLI/字符串到调度定义的解析工具
# ==========================

def _normalize_points(obj: Any) -> Optional[List[Point]]:
    """将任意类 list/tuple 的对象规范化为 [(t, v), ...] 形式。
    - 支持 [[t, v], ...] | [(t, v), ...]
    - 自动转换为 float
    - 不合法则返回 None
    """
    try:
        pts = []
        for p in obj:  # type: ignore[assignment]
            if isinstance(p, (list, tuple)) and len(p) == 2:
                t, v = float(p[0]), float(p[1])
                pts.append((t, v))
            else:
                return None
        return pts if len(pts) > 0 else None
    except Exception:
        return None


def _parse_points_str(s: str) -> Optional[List[Point]]:
    """解析分号分隔的点串：
    允许以下形式（空格可选）：
      - "0,1;0.5,0.5;1,0.1"
      - "0:1; 0.5:0.5; 1:0.1"
    返回 [(0.0,1.0), (0.5,0.5), (1.0,0.1)] 或 None
    """
    s = s.strip()
    if not s:
        return None
    pairs = [p for p in s.split(';') if p.strip()]
    if not pairs:
        return None
    out: List[Point] = []
    for p in pairs:
        p = p.strip()
        if ',' in p:
            t_str, v_str = p.split(',', 1)
        elif ':' in p:
            t_str, v_str = p.split(':', 1)
        else:
            return None
        try:
            t = float(t_str.strip())
            v = float(v_str.strip())
        except ValueError:
            return None
        out.append((t, v))
    return out if out else None


def parse_schedule_def(spec: Any) -> Optional[ScheduleDef]:
    """将多种规格（来自命令行/配置）解析为统一的调度定义。

    支持输入：
    - None => None
    - number => 常数
    - list/tuple => [(t, v), ...]（或可直接 eval_schedule)
    - callable => 原样返回
    - str => 依次尝试：
        * 忽略大小写的 'none' => None
        * 'const:<number>' => 常数
        * ast.literal_eval() => 解析 Python 字面量（安全）：
            - 单个数字 (float/int)
            - 列表/元组的二元组序列，如 [(0,1.0),(1,0.1)] 或 ((0,1.0),(1,0.1))
        * 分号分隔点串："0,1;0.5,0.5;1,0.1" 或 "0:1;0.5:0.5;1:0.1"

    返回：
    - 可直接被 eval_schedule 接受的类型：number / [(t,v), ...] / callable
    - None 表示不设置

    示例：
    - "1.0" -> 1.0
    - "const:0.5" -> 0.5
    - "[(0,1.0),(0.5,0.5),(1.0,0.1)]" -> [(0.0,1.0),(0.5,0.5),(1.0,0.1)]
    - "0,1;0.5,0.5;1,0.1" -> [(0.0,1.0),(0.5,0.5),(1.0,0.1)]
    """
    # 直接透传常见类型
    if spec is None:
        return None
    if callable(spec):
        return spec  # type: ignore[return-value]
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (list, tuple)):
        norm = _normalize_points(spec)
        return norm if norm is not None else None

    # 字符串解析
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return None
        low = s.lower()
        if low == 'none' or low == 'null':
            return None
        if low.startswith('const:'):
            try:
                return float(s.split(':', 1)[1].strip())
            except Exception:
                return None
        # 单个数字字符串
        try:
            return float(s)
        except ValueError:
            pass
        # 尝试 Python 字面量安全解析
        try:
            lit = ast.literal_eval(s)
            if isinstance(lit, (int, float)):
                return float(lit)
            norm = _normalize_points(lit)
            if norm is not None:
                return norm
        except Exception:
            pass
        # 尝试分号分隔的点串
        pts = _parse_points_str(s)
        if pts is not None:
            return pts
        return None

    # 其他类型不支持
    return None

