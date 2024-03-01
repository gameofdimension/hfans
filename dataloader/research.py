import os
import time

import torch
import torch.distributed as dist


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, interval_sec):
        self.len = length
        self.data = torch.arange(length)
        self.interval_sec = interval_sec

    def __getitem__(self, index):
        if self.interval_sec > 0:
            time.sleep(self.interval_sec)
        return self.data[index], os.getpid()

    def __len__(self):
        return self.len


def main():
    """
    1. 验证了异步多进程加载其实粒度很粗，一个 batch 来自于一个 producer 进程，可以从 pid 信息看出
    2. 还有另一个后果是，如果各进程的时间上没有错开的话，num_workers 个批次会同时到达，
        事实上会造成一个 batch_size*num_workers 的一个大批次，其耗时是 batch_size 倍的单条数据处理耗时
    """
    init_distributed()

    # dataset = Dataset(10000, 0)
    dataset = Dataset(10000, 1)
    # dataset = Dataset(10000, 2)
    # dataset = Dataset(10000, 10)
    batch_size = 16
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False),
        drop_last=True,
        num_workers=4,
    )

    start = time.time()
    for v, pid in dataloader:
        print(v, pid, (time.time()-start)*1000)
        start = time.time()

    cleanup()


# torchrun --nnodes=1 --nproc-per-node=1 --master-addr=localhost
# --master-port=30601 --node-rank=0 -m dataloader.research
if __name__ == '__main__':
    main()
