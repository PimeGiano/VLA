#-*-coding:utf-8-*-

# reference to ~/miniconda3/envs/RL4VLA_LIBERO/lib/python3.10/site-packages/robosuite/renderers/context/egl_context.py

import subprocess
from termcolor import cprint

from .egl_extent import get_device_uuids

def get_CUDA_info():
    """ 获取 nvidia-smi 的输出并解析 GPU 设备信息 """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid', '--format=csv,noheader'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return {}, []

    nvidia_devices = {}  # UUID -> GPU Index 映射
    gpu_list = []  # 按索引顺序存储GPU信息
    cuda_uuid = []
    
    print("NVIDIA GPU Information from nvidia-smi:")
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
            
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 3:
            index = parts[0]
            name = parts[1]
            uuid = parts[2]
            cuda_uuid.append(normalize_uuid(uuid))
            
            try:
                gpu_index = int(index)
                gpu_info = {
                    'index': gpu_index,
                    'name': name,
                    'uuid': uuid
                }
                gpu_list.append(gpu_info)
                nvidia_devices[uuid] = gpu_index  # UUID 映射到设备 ID
                
                print(f"  GPU {index}: {name}")
                print(f"    UUID: {uuid}")
            except ValueError:
                print(f"Warning: Invalid GPU index '{index}' in line: {line}")
    
    print(f"\nTotal GPUs found: {len(gpu_list)}")

    return cuda_uuid


def normalize_uuid(uuid_string):
    """
    将UUID字符串标准化，去掉GPU-前缀和所有连字符
    例如: 'GPU-0c65ddfc-9c41-a04e-62a6-53312f768144' -> '0c65ddfc9c41a04e62a653312f768144'
    """
    if uuid_string.startswith('GPU-'):
        uuid_string = uuid_string[4:]  # 去掉'GPU-'前缀

    # 去掉所有连字符
    normalized = uuid_string.replace('-', '')
    return normalized.lower()  # 转为小写保持一致性


def get_EGL_info():
    # 这个就是env创建的时候用到的顺序
    device_uuids = get_device_uuids()
    egl_uuid = []
    for uuid in device_uuids:
        egl_uuid.append(uuid.hex())

    return egl_uuid


def uuid_align():
    cuda_uuid = get_CUDA_info()
    egl_uuid = get_EGL_info()

    print(f"cuda_uuid: \n {cuda_uuid}")
    print(f"egl_uuid: \n {egl_uuid}")

    correspond_cuda2egl = {}
    for i, uuid in enumerate(cuda_uuid):
        if uuid in egl_uuid:
            correspond_cuda2egl[i] = egl_uuid.index(uuid)
            cprint(f"cuda {i} -> egl {egl_uuid.index(uuid)}", "red")
        else:
            cprint(f"cuda {i} -> egl None", "red")

    return correspond_cuda2egl

# 测试nvidia-smi功能
if __name__ == "__main__":
    uuid_align()