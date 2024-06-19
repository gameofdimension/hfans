import torch


@torch.compile
def fn(x, n):
    y = x ** 2
    if n >= 0:
        return (n + 1) * y
    else:
        return y / n


x = torch.randn(200)
fn(x, 2)
fn(x, 3)
fn(x, -2)
