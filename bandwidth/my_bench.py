import os
import sys

import torch

try:
    import torch_npu  # type: ignore # noqa
    from torch_npu.contrib import transfer_to_npu  # noqa # type: ignore
except ImportError:
    pass
import torch.distributed as dist
from torch.utils import benchmark


def init_distributed():

    # Initializes the distributed backend
    # which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=rank
    )
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    # dist.barrier()
    return world_size, rank, local_rank


def run_all_reduce(world_size, tensor):
    dist.all_reduce(tensor)


def run_all_gather(world_size, tensor, target):
    dist.all_gather(target, tensor)


def is_main_process():
    try:
        if dist.get_rank() == 0:
            return True
        else:
            return False
    except Exception:
        return True


def profile_model(fn, min_run_time=5):
    # https://github.com/facebookresearch/xformers/issues/678
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    res = benchmark.Timer(
        stmt='fn()',
        globals={"fn": fn},
        label="profile",
        sub_label="",
        description=""
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2 ** 20
    # memory = f"Memory used: {memory} MB"
    # print(res)
    # print(memory)
    return memory, res


def main():
    init_distributed()
    world_size = int(os.environ["WORLD_SIZE"])

    method = sys.argv[1]
    assert method in ['all_reduce', 'all_gather']

    if is_main_process():
        cols = ["Volume", "Mean", "P25", "Median", "P75", "Bandwidth"]
        print(
            f"{cols[0]:>6}/MB"
            f"{cols[1]:>12}/us"
            f"{cols[2]:>12}/us"
            f"{cols[3]:>12}/us"
            f"{cols[4]:>12}/us"
            f"{cols[5]:>12}/GB/s"
        )
    for power in range(20, 29):
        tensor_size = 2**power
        if method == 'all_reduce':
            tensor = torch.randn(tensor_size, device='cuda')
            memory, res = profile_model(
                lambda: run_all_reduce(world_size, tensor))
        elif method == 'all_gather':
            tensor = torch.randn(tensor_size, device='cuda')
            target = [torch.zeros_like(tensor) for _ in range(world_size)]
            memory, res = profile_model(
                lambda: run_all_gather(world_size, tensor, target))

        if is_main_process():
            print(
                f"{4*tensor_size/(2**20):>9.1f}"
                f"{res.mean*10**6:>15.3f}"
                f"{res._p25*10**6:>15.3f}"
                f"{res.median*10**6:>15.3f}"
                f"{res._p75*10**6:>15.3f}"
                f"{(4*tensor_size/1e9)/res.mean:>17.3f}"
            )
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
