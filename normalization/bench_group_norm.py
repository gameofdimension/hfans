import torch
from torch import nn
from torch.utils import benchmark

from apex.normalization.fused_layer_norm import FusedLayerNorm
from apex.contrib.group_norm import GroupNorm as ApexGroupNorm


def profile_model(fn, desc, min_run_time=5):
    # https://github.com/facebookresearch/xformers/issues/678
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    res = benchmark.Timer(
        stmt='fn()',
        globals={"fn": fn},
        label="profile",
        sub_label="",
        description=f"{desc}"
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2 ** 20
    memory = f"Memory used: {memory} MB"
    print(res)
    print(memory)


def main():
    param = {
        'device': 'cuda',
        'dtype': torch.bfloat16,
    }
    data = torch.randn(
        4, 128, 1024, 768, **param).to(
            memory_format=torch.channels_last)  # type: ignore

    gn = nn.GroupNorm(32, 128).to(
        memory_format=torch.channels_last, **param)  # type: ignore
    profile_model(lambda: gn(data), desc='pytorch group norm')

    apex_gn = ApexGroupNorm(32, 128, **param)
    profile_model(lambda: apex_gn(data), desc='apex group norm')


if __name__ == '__main__':
    main()
