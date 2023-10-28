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

    def get_cos_sin(pos: int):
        return cos[pos], sin[pos]

    return get_cos_sin


def apply_rotary(vector: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    assert vector.dim() == 2
    assert cos.dim() == sin.dim() == 1
    assert cos.size(-1) == sin.size(-1) == vector.size(-1)
    cos = cos.view(1, -1)
    sin = sin.view(1, -1)
    nh, d = vector.size()
    tmp = torch.cat([vector[:, d // 2:], vector[:, :d // 2]], dim=1)
    return vector * cos + tmp * sin


if __name__ == '__main__':
    factory = precompute_cos_sin(100, 64)

    vec = torch.randn(50, 64)
    for i in range(100):
        cos, sin = factory(i)
        apply_rotary(vec, cos, sin)
