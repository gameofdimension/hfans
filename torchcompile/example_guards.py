import torch


@torch.compile
def fn(a, b):
    return a * len(b)


fn(torch.arange(10), "Hello")
fn(torch.arange(10), "Hi")
