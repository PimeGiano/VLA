```shell
conda create -n FAN-VLA -y python=3.10
conda activate FAN-VLA

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

cd openvla && pip install -e . && cd ..

pip install -U tyro 
pip install datasets==3.3.2
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..
```

```shell
# Download weights
pip install -U huggingface_hub hf-transfer
huggingface-cli download --resume-download gen-robot/openvla-7b-rlvla-warmup --local-dir ./weights/openvla-7b-rlvla-warmup
```

# run

```shell
# ./openvla_rl_ms_train.sh [cuda_id] [task_type:1-RL4VLA 2-OpenVLA + PPO 3-OpenVLA + FAN-PPO]
./openvla_rl_ms_train.sh 0 3
```

