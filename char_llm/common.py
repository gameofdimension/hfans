import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        val = hidden_states * hidden_states
        norm = torch.sqrt(torch.mean(val, dim=-1, keepdim=True) + self.eps)
        return hidden_states / norm * self.weight


def precompute_cos_sin(n: int, d: int):
    assert d > 0 and d % 2 == 0

    base = torch.tensor(10000)
    cos = torch.zeros(n, d, requires_grad=False)
    sin = torch.zeros(n, d, requires_grad=False)
    for i in range(n):
        for j in range(d // 2):
            theta = base ** (-2 * j / d)
            cos[i, j] = torch.cos(i * theta)
            cos[i, j + d // 2] = torch.cos(i * theta)
            sin[i, j] = -torch.sin(i * theta)
            sin[i, j + d // 2] = torch.sin(i * theta)

    def get_cos_sin(device):
        return cos.to(device), sin.to(device)

    return get_cos_sin


def apply_rotary(vector: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    assert vector.dim() == 4
    assert cos.dim() == sin.dim() == 2
    assert cos.size(-1) == sin.size(-1) == vector.size(-1)
    assert cos.size(0) == sin.size(0) == vector.size(0)
    sl, bs, nh, d = vector.size()
    cos = cos.view(sl, 1, 1, -1)
    sin = sin.view(sl, 1, 1, -1)
    tmp = torch.cat([vector[..., d // 2:], vector[..., :d // 2]], dim=-1)
    return vector * cos + tmp * sin
