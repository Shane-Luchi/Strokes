import torch
import torch.distributed as dist
import os
import time

def setup_distributed():
    """初始化分布式环境"""
    # 当使用 torchrun 时，RANK, WORLD_SIZE, LOCAL_RANK 会由torchrun自动设置
    # 对于GPU通信，后端通常使用 'nccl'
    if not dist.is_initialized(): # 确保只初始化一次
        dist.init_process_group(backend='nccl')

    # LOCAL_RANK 是由 torchrun 为每个进程设置的环境变量，通常表示当前进程应使用的GPU的本地索引
    # 例如，如果 CUDA_VISIBLE_DEVICES="2,3"，那么GPU 2对进程来说是local_rank 0, GPU 3是local_rank 1
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank) # 关键：为当前进程设置默认GPU
    
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Process Init] Global Rank: {global_rank}, Local Rank: {local_rank}, World Size: {world_size}, Device: cuda:{local_rank}")
    return global_rank, local_rank, world_size

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    global_rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") # 每个进程操作自己的GPU

    # 1. 在每个GPU上创建一个张量
    # 张量的值与全局排名相关，方便观察
    initial_value = float(global_rank + 1)
    tensor_on_gpu = torch.tensor([initial_value, initial_value * 2], dtype=torch.float32, device=device)
    
    print(f"[Rank {global_rank} on cuda:{local_rank}] Initial tensor: {tensor_on_gpu}")

    # 2. 执行一个简单的分布式操作：all_reduce (求和)
    # 每个进程贡献自己的张量，操作完成后，所有进程的张量都会变成所有原始张量之和
    if world_size > 1: # 只有在多于一个进程时，分布式操作才有意义
        dist.all_reduce(tensor_on_gpu, op=dist.ReduceOp.SUM)
        print(f"[Rank {global_rank} on cuda:{local_rank}] Tensor after all_reduce (SUM): {tensor_on_gpu}")
    else:
        print(f"[Rank {global_rank} on cuda:{local_rank}] Single process, skipping all_reduce.")

    # 3. （可选）使用 barrier 等待所有进程到达此处，确保打印信息不混乱
    if world_size > 1:
        print(f"[Rank {global_rank} on cuda:{local_rank}] Waiting at barrier...")
        dist.barrier()
        print(f"[Rank {global_rank} on cuda:{local_rank}] Passed barrier.")

    # 为了方便查看日志，稍微错开一点退出时间
    time.sleep(1 + global_rank * 0.1)

    cleanup_distributed()
    print(f"[Rank {global_rank} on cuda:{local_rank}] Cleaned up and exiting.")

if __name__ == "__main__":
    main()