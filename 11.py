import os
print(f"Initial CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}") # 检查外部是否已设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 强制设置
print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch
print(f"PyTorch version: {torch.__version__}")

try:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # 在这里，device_count() 应该返回 1
        print(f"CUDA device count: {torch.cuda.device_count()}")
        # current_device() 应该返回 0
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name for device 0: {torch.cuda.get_device_name(0)}") # 明确指定设备0

        # 尝试在设备0上做简单操作
        x = torch.tensor([1.0, 2.0]).to('cuda:0')
        print(f"Tensor x on cuda:0: {x}")
        print("Simple CUDA test successful on device 0.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()