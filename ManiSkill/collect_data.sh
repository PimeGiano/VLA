#!/bin/bash

# 检查参数数量是否正确
if [ $# -ne 2 ]; then
  echo "用法: sh collect_data.sh <cuda设备号> <命令编号(1或2)>"
  echo "示例: sh collect_data.sh 7 2"
  exit 1
fi

# 从命令行参数获取CUDA设备和命令编号
cuda=$1
command_num=$2

# 验证命令编号是否有效
if [ "$command_num" -ne 1 ] && [ "$command_num" -ne 2 ]; then
  echo "错误: 命令编号必须是1或2"
  exit 1
fi

# 执行选中的命令
if [ "$command_num" -eq 1 ]; then
  echo "开始执行OpenVLA warm-up数据收集 (CUDA设备: $cuda)..."
  # OpenVLA warm-up数据收集（额外5条轨迹用于性能评估）
  CUDA_VISIBLE_DEVICES=$cuda \
  python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
    -e "PutOnPlateInScene25Single-v1" \
    --save_video --save_data --num_procs 1 --num_traj 75 --seed=0
else
  echo "开始执行SFT数据收集 (CUDA设备: $cuda)..."
  # SFT数据收集（额外16条轨迹用于性能评估）
  CUDA_VISIBLE_DEVICES=$cuda \
  python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
    -e "PutOnPlateInScene25Main-v3" \
    --save_video --save_data --num_procs 16 --num_traj 16400 --seed=100
fi

# 检查命令执行结果
if [ $? -ne 0 ]; then
  echo "命令执行失败"
  exit 1
fi

echo "数据收集任务已完成"
