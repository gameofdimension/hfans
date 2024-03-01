import os
import time

import torch


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

    # dataset = Dataset(10000, 0)
    dataset = Dataset(10000, 1)
    # dataset = Dataset(10000, 2)
    # dataset = Dataset(10000, 10)
    batch_size = 16
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
    )

    start = time.time()
    for v, pid in dataloader:
        print(f"batch time {(time.time()-start)*1000:.3f}", v, pid)
        start = time.time()


# python -m dataloader.local
if __name__ == '__main__':
    main()
