import torch
from apex.contrib.group_norm import GroupNorm as ApexGroupNorm
from flash_attn import flash_attn_func
from torch import nn
from tqdm import tqdm


class FlashModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.permute(0, 2, 1).unsqueeze(2).contiguous()
        k = k.permute(0, 2, 1).unsqueeze(2).contiguous()
        v = v.permute(0, 2, 1).unsqueeze(2).contiguous()

        out: torch.Tensor = flash_attn_func(q, k, v)  # type: ignore
        out = out.squeeze(2).permute(0, 2, 1).contiguous()
        return out


def flash_compile():
    param = {
        'device': 'cuda',
        'dtype': torch.bfloat16,
        'requires_grad': True,
    }
    model = FlashModule().to(device=torch.device('cuda'), dtype=torch.bfloat16)
    model = torch.compile(model=model)
    for i in tqdm(range(100)):
        q = torch.randn(4, 256, 256*192, **param)
        k = torch.randn(4, 256, 256*192, **param)
        v = torch.randn(4, 256, 256*192, **param)

        out = model(q, k, v)
        loss = out.sum()
        loss.backward()


def apex_group_norm_compile():
    group = 32
    channels = 512
    h, w = 256, 192
    model = ApexGroupNorm(group, channels, act='silu',
                          device='cuda', dtype=torch.bfloat16)
    model = torch.compile(model=model)

    param = {
        'device': 'cuda',
        'dtype': torch.bfloat16,
        'requires_grad': True,
    }
    grad = torch.randn(
        4, channels, h, w, **param).to(
        memory_format=torch.channels_last)  # type: ignore

    for i in tqdm(range(100)):
        data = torch.randn(
            4, channels, h, w, **param).to(
                memory_format=torch.channels_last)  # type: ignore
        print(data.is_contiguous(memory_format=torch.channels_last))
        out = model(data)
        # print(out.shape, out.stride())
        print('xxx', out.is_contiguous(memory_format=torch.channels_last))
        y = out*out
        print(y.shape, y.stride())
        # loss = y.mean()
        # loss.backward()
        y.backward(grad)


if __name__ == '__main__':
    # flash_compile()
    apex_group_norm_compile()
