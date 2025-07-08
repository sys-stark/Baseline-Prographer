import torch

# 检查CUDA是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda_available}")

if is_cuda_available:
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    # 获取当前GPU设备名称
    current_gpu_name = torch.cuda.get_device_name(0)
    print(f"Current GPU Name: {current_gpu_name}")