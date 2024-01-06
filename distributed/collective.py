import os
import torch
import torch.distributed as dist
from loguru import logger


def cleanup():
    dist.destroy_process_group()


def init_distributed():

    # Initializes the distributed backend
    # which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(backend="nccl",
                            init_method=dist_url,
                            world_size=world_size,
                            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    return world_size, rank, local_rank


def all_reduce(rank: int):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    tensor = tensor.to('cuda')
    logger.info(f"before all_reduce {tensor}")
    dist.all_reduce(tensor)  # default op is sum
    logger.info(f"after all_reduce {tensor}")
    dist.barrier()


def all_gather(world_size: int, rank: int):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    tensor = tensor.to('cuda')
    container = [torch.zeros_like(tensor) for _ in range(world_size)]
    logger.info(f"before all_gather {tensor}, {container}")
    dist.all_gather(container, tensor)  # default op is sum
    logger.info(f"after all_gather {container}")
    dist.barrier()


def scatter(world_size: int, rank: int):
    output = torch.zeros(2, device='cuda')
    if dist.get_rank() == 0:
        data = [torch.ones(2, device='cuda') + i for i in range(world_size)]
    else:
        data = None
    logger.info(f"before scatter {output}")
    dist.scatter(output, data, src=0)
    logger.info(f"after scatter {output}")
    dist.barrier()


def broadcast_from(rank: int):
    if dist.get_rank() == rank:
        tensor = torch.tensor([42.5, 9.36], device='cuda')
    else:
        tensor = torch.tensor([1.0, 1.0], device='cuda')
    logger.info(f"before broadcast {tensor}")
    dist.broadcast(tensor, src=rank)
    logger.info(f"after broadcast {tensor}")
    dist.barrier()


def send_recv(world_size: int):
    if dist.get_rank() == 0:
        tensor = torch.tensor([111.111], device='cuda')
        for i in range(world_size):
            if i == 0:
                continue
            dist.send(tensor * i, i)
    else:
        tensor = torch.tensor([0.0], device='cuda')
        logger.info(f"before recv {tensor}")
        dist.recv(tensor, src=0)
        logger.info(f"before recv {tensor}")
    dist.barrier()


def main():
    world_size, rank, local_rank = init_distributed()

    logger.remove(0)
    logger.add(f"dist_log_{os.getpid()}.txt", level="DEBUG")

    all_reduce(rank)
    all_gather(world_size, rank)

    scatter(world_size, rank)

    broadcast_from(1)

    send_recv(world_size)

    cleanup()


# torchrun --nnodes=1 --nproc-per-node=3 --master-addr=localhost
# --master-port=30601 --node-rank=0 -m distributed.collective
if __name__ == '__main__':
    main()
