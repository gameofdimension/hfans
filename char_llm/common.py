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


class Rotary:
    def __init__(self, d: int, paper: bool = False):
        assert d % 2 == 0
        self.d = d
        self.matrix_lst = []
        self.paper = paper

    def _pad(self, target: int):
        base = 10000
        for m in range(len(self.matrix_lst), target + 1):
            matrix = torch.zeros(size=(self.d, self.d))

            for j in range(self.d // 2):
                theta = base ** (-2 * j / self.d)
                # 以下是论文实现
                if self.paper:
                    matrix[2 * j, 2 * j] = math.cos(m * theta)
                    matrix[2 * j, 2 * j + 1] = -math.sin(m * theta)
                    matrix[2 * j + 1, 2 * j + 1] = math.cos(m * theta)
                    matrix[2 * j + 1, 2 * j] = math.sin(m * theta)
                # 以下是 llama 实现
                else:
                    matrix[j, j] = math.cos(m * theta)
                    matrix[j, j + self.d // 2] = -math.sin(m * theta)
                    matrix[j + self.d // 2, j + self.d // 2] = math.cos(m * theta)
                    matrix[j + self.d // 2, j] = math.sin(m * theta)
            self.matrix_lst.append(matrix)

    def apply(self, m: int, vec: torch.Tensor):
        assert m >= 0
        assert vec.size(-1) == self.d
        if m >= len(self.matrix_lst):
            self._pad(m)
        matrix = self.matrix_lst[m]
        return matrix @ vec


if __name__ == '__main__':
    ro = Rotary(d=64)

    factory = precompute_cos_sin(100, 64)
    vec = torch.randn(50, 64)
    for i in range(100):
        cos, sin = factory(i)

        out1 = torch.zeros_like(vec)
        for k in range(vec.size(0)):
            out1[k] = ro.apply(i, vec[k])

        out2 = apply_rotary(vec, cos, sin)
        assert torch.max(torch.abs(out1 - out2)).item() < 1e-4
