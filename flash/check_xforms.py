import math
from typing import Optional
import torch as th
import numpy as np
from torch.utils import benchmark
from xformers.ops import memory_efficient_attention


def gold(q: th.Tensor, k: th.Tensor, v: th.Tensor, ch: int,
         grad: Optional[th.Tensor], with_back: bool):
    scale = 1 / math.sqrt(math.sqrt(ch))
    # More stable with f16 than dividing afterwards
    weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
    weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
    out = th.einsum("bts,bcs->bct", weight, v)
    if with_back:
        assert out.size() == grad.size()
        out.backward(grad)
    return out


def xformers(q: th.Tensor, k: th.Tensor, v: th.Tensor, ch: int,
             grad: Optional[th.Tensor], with_back: bool):
    scale = 1 / math.sqrt(ch)
    out = memory_efficient_attention(q, k, v, scale=scale)
    if with_back:
        assert out.size() == grad.size()
        out.backward(grad)
    return out


def xformers_axis_switch(q: th.Tensor, k: th.Tensor, v: th.Tensor, ch: int,
                         grad: th.Tensor, with_back: bool):
    q = q.permute(0, 2, 1).contiguous()
    k = k.permute(0, 2, 1).contiguous()
    v = v.permute(0, 2, 1).contiguous()
    scale = 1 / math.sqrt(ch)
    out = memory_efficient_attention(q, k, v, scale=scale).permute(0, 2, 1)
    if with_back:
        assert out.size() == grad.size()
        out.backward(grad)
    return out


def relative_error(a: th.Tensor, b: th.Tensor):
    indcies = th.argmax((a-b).abs())
    max_index = np.unravel_index(indcies.cpu().numpy(), shape=a.size())
    v1 = a[max_index]
    v2 = b[max_index]
    return 2*(v1-v2).abs()/(v1.abs()+v2.abs())


def profile_model(fn, min_run_time=5):
    # https://github.com/facebookresearch/xformers/issues/678
    th.cuda.reset_peak_memory_stats()
    th.cuda.synchronize()
    res = benchmark.Timer(
        stmt='fn()',
        globals={"fn": fn},
        label="profile",
        sub_label="",
        description=""
    ).blocked_autorange(min_run_time=min_run_time)
    th.cuda.synchronize()
    memory = th.cuda.max_memory_allocated() / 2 ** 20
    memory = f"Memory used: {memory} MB"
    print(res)
    print(memory)


def do_profile(dtype, with_back: bool):
    print(f"###### profile with dtype: {dtype}, backward: {with_back}")
    shape = [64, 256, 1024]
    ch = shape[-2]
    q = th.randn(shape, requires_grad=with_back).cuda().to(dtype)
    k = th.randn(shape, requires_grad=with_back).cuda().to(dtype)
    v = th.randn(shape, requires_grad=with_back).cuda().to(dtype)

    print("-"*100)
    print("official:")
    grad = th.randn_like(v, requires_grad=with_back)
    profile_model(lambda: gold(q, k, v, ch, grad, with_back))

    print("-"*100)
    print("with axis switch:")
    grad = th.randn_like(v, requires_grad=with_back)
    profile_model(lambda: xformers_axis_switch(q, k, v, ch, grad, with_back))

    print("-"*100)
    print("without axis switch:")
    q = q.permute(0, 2, 1).contiguous()
    k = k.permute(0, 2, 1).contiguous()
    v = v.permute(0, 2, 1).contiguous()
    grad = th.randn_like(v, requires_grad=with_back)
    profile_model(lambda: xformers(q, k, v, ch, grad, with_back))


def check_error(dtype):
    print(f"###### check error with dtype {dtype}")
    shape = [16, 256, 1024]
    q = th.randn(shape).cuda().to(dtype)
    k = th.randn(shape).cuda().to(dtype)
    v = th.randn(shape).cuda().to(dtype)

    ch = shape[-2]
    a = gold(q, k, v, ch, grad=None, with_back=False)
    b = xformers(q.transpose(-2, -1).contiguous(),
                 k.transpose(-2, -1).contiguous(),
                 v.transpose(-2, -1).contiguous(),
                 ch, grad=None, with_back=False).transpose(-2, -1)
    print(a.dtype, b.dtype)

    print(f"relative error: {relative_error(a,b):.6f}")


if __name__ == '__main__':
    # check_error(th.float32)
    # check_error(th.float16)

    # do_profile(th.float32, False)
    # do_profile(th.float16, False)

    do_profile(th.float32, True)
    do_profile(th.float16, True)
