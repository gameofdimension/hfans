#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: MIT
#

# adapted from https://github.com/IBM/pytorch-communication-benchmarks/blob/main/allreduce-loop.py

import os
import sys
import torch
try:
    import torch_npu  # type: ignore # noqa
    from torch_npu.contrib import transfer_to_npu  # noqa # type: ignore
except ImportError:
    pass
import torch.distributed as dist
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--multiplier", type=int, default=1)

args = parser.parse_args()
multiplier = args.multiplier

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)

dist.init_process_group("nccl")

# if rank == 0:
#     print("NCCL version : ", torch.cuda.nccl.version(), file=sys.stderr)

torch.manual_seed(1235911);

nMB = 10000.0
npts = int(nMB*1.0e6/4.0)

Tensor = torch.rand(npts, device='cuda')
torch.cuda.synchronize()

if rank == 0:
    print(" size(MB)   tavg(usec)    tmin(usec)    tmax(usec)  avgbw(GB/sec)  maxbw(GB/sec)  minbw(GB/sec)", file=sys.stderr)

for nMB in [0.10,0.12,0.15,0.20,0.32,0.40,0.50,0.64,0.80,1.00,1.25,1.50,2.00,3.16,4.00,5.00,6.40,8.00,\
            10.0,12.5,15.0,20.0,31.6,40.0,50.0,64.0,80.0,100.0,125.0,160.0,200.0,250.0,316.0,400.0,500.0,640.0,800.0,\
            1000.0,1250.0,1600.0,2000.0,2500.0,3160.0,4000.0,5000.0,6400.0,8000.0]:

    if nMB < 10.0:
        maxiter = 100*multiplier
    elif nMB < 512.0:
        maxiter = 20*multiplier
    elif nMB < 2000.0:
        maxiter = 10*multiplier
    else:
        maxiter = 5*multiplier

    npts = int(nMB*1.0e6/4.0)
    nm1 = int(npts - 1)

    # launch two calls outside the timing loop
    dist.all_reduce(Tensor[0:nm1], op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.all_reduce(Tensor[0:nm1], op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    tbeg = time.perf_counter()
    t1 = tbeg
    tmin = 1.0e30
    tmax = 0.0

    for i in range(maxiter):
        dist.all_reduce(Tensor[0:nm1], op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if (t2 - t1) < tmin:
            tmin = (t2 - t1)
        if (t2 - t1) > tmax:
            tmax = (t2 - t1)
        t1 = t2

    torch.cuda.synchronize()
    tend = time.perf_counter()

    elapsed = tend - tbeg
    tavg = elapsed / maxiter

    # avgbw = 4.0*2.0e-9*npts*((world_size - 1)/world_size)/tavg
    # maxbw = 4.0*2.0e-9*npts*((world_size - 1)/world_size)/tmin
    # minbw = 4.0*2.0e-9*npts*((world_size - 1)/world_size)/tmax
    avgbw = 4.0*1.0e-9*npts/tavg
    maxbw = 4.0*1.0e-9*npts/tmin
    minbw = 4.0*1.0e-9*npts/tmax

    if rank == 0:
        print("{:8.2f}".format(nMB), "  ", "{:7.1f}".format(tavg*1.0e6), "      ", "{:7.1f}".format(tmin*1.0e6), "      ", "{:7.1f}".format(tmax*1.0e6), \
              "     ", "{:7.2f}".format(avgbw), "      ", "{:7.2f}".format(maxbw), "      ", "{:7.2f}".format(minbw), file=sys.stderr)

dist.destroy_process_group()