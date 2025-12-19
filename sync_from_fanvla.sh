#!/bin/bash
SOURCE_DIR="/home/zyz518348/workspace/zyz518348/FAN-VLA"
TARGET_DIR="/home/zyz518348/workspace/zyz518348/backup"

echo "从 FAN-VLA 同步干净代码到 backup"

# 使用rsync，排除所有数据文件和编译文件
rsync -av \
  --exclude='*__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.vscode' \
  --exclude='.idea' \
  --exclude='wandb/' \
  --exclude='weights/' \
  --exclude='checkpoints/' \
  --exclude='ms3_datasets/' \
  --exclude='.git' \
  "$SOURCE_DIR/" "$TARGET_DIR/"

# 确保.gitignore被复制
if [ -f "$SOURCE_DIR/.gitignore" ]; then
  cp "$SOURCE_DIR/.gitignore" "$TARGET_DIR/"
fi

echo "同步完成！"
