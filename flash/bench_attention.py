import math

import torch


def naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # [b,h,s,d]
    assert q.size(-1) == k.size(-1) == v.size(-1)
    scale = 1 / math.sqrt(q.size(-1))
    k = k.transpose(-1, -2)
    s = q @ k
    s = s * scale
    p = torch.nn.functional.softmax(s, dim=-1)
    out = p @ v
    return out


def cpp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    scale = 1 / math.sqrt(q.size(-1))
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # Expected query, key and value to all be of dtype: {Half, BFloat16}.
    # Got Query dtype: float, Key dtype: float, and Value dtype: float instead.
    q = q.half()
    k = k.half()
    v = v.half()
    scale = 1 / math.sqrt(q.size(-1))
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def xformer(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    scale = 1 / math.sqrt(q.size(-1))
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=False, enable_mem_efficient=True
    ):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def main():
    shape = (5, 6, 7, 512)
    q = torch.randn(*shape).to('cuda')
    k = torch.randn(*shape).to('cuda')
    v = torch.randn(*shape).to('cuda')

    x = naive(q, k, v)
    y = cpp(q, k, v)
    z = flash(q, k, v)
    u = xformer(q, k, v)
    print((y - z).abs().max())
    print((x - z).abs().max())
    print((x - y).abs().max())
    print((x - u).abs().max())


def fp16error():
    a = torch.randn(10, 10).half()
    b = a-0.2+0.3
    c = a+0.3-0.2
    print((b-c).abs().max())


def fp32to16():
    a = torch.tensor([1, 2, -1, 65503, 65504, 65504.1, 65504.5, 65505,
                      65506, 66000, 67000, -65506, -66000]).to('cuda')
    print(a)
    print(a.half())


if __name__ == '__main__':
    # main()
    fp16error()
    fp32to16()
