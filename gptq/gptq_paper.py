import random

import torch


def gen_symmetric_matrix(n: int):
    a = torch.randn(n, n)
    return a @ a.T


def remove_row_col(m: torch.Tensor, idx: int):
    m = torch.cat([m[:idx, :], m[idx + 1:, :]], dim=0)
    m = torch.cat([m[:, :idx], m[:, idx + 1:]], dim=1)
    return m


def test_cholesky():
    """
    测试 cholesky 分解
    :return:
    """
    m = gen_symmetric_matrix(10)
    ch = torch.linalg.cholesky(m)
    assert torch.allclose(ch @ ch.T, m)


def test_cholesky_inverse():
    m = gen_symmetric_matrix(10)
    ch = torch.linalg.cholesky(m)
    ch_inv = torch.cholesky_inverse(ch)
    assert torch.allclose(m @ ch_inv, torch.eye(10), atol=1e-4)


def naive(h: torch.Tensor, idx: int):
    """
    h 矩阵 去掉 idx 行和 idx 列，再求逆
    :param idx:
    :param h:
    :return:
    """
    hm = remove_row_col(h, idx)
    assert h.size(0) == hm.size(0) + 1
    assert torch.allclose(hm, hm.T)

    ihm = torch.linalg.inv(hm)
    return hm, ihm


def smart(h: torch.Tensor, invh: torch.Tensor, idx: int):
    """
    h 矩阵 去掉 idx 行和 idx 列，再求逆
    :param h:
    :param invh:
    :param idx:
    :return:
    """
    assert h.size() == invh.size()
    assert torch.allclose(h @ invh, torch.eye(h.size(0)), atol=1e-3), torch.max(
        torch.abs(h @ invh - torch.eye(h.size(0))))
    out = invh - invh[:, idx].view(-1, 1) @ invh[idx, :].view(1, -1) / invh[idx, idx]
    return remove_row_col(out, idx)


def test_remove_inverse():
    """
    问题来自 gptq 论文。一个矩阵去掉 q 行和 q 列，再求逆，这里验证高效算法的正确性
    :return:
    """
    h = gen_symmetric_matrix(10)
    invh = torch.linalg.inv(h)
    for _ in range(10):
        i = random.randint(0, h.size(0) - 1)
        hm, ihm = naive(h, i)
        assert torch.allclose(hm @ ihm, torch.eye(9), atol=1e-4)

        out = smart(h, invh, i)
        assert torch.allclose(hm @ out, torch.eye(9), atol=1e-4)


def test_cholesky_equivalence():
    """
    we check the claim from paper as below:

    "Indeed, the row removal via (3) for our symmetric H−1 essentially corresponds to
    taking a Cholesky decomposition, except for the minor difference that the latter
    divides row q by ([H−1 Fq ]qq)1/2."

    :return:
    """
    h = gen_symmetric_matrix(10)
    invh = torch.linalg.inv(h)

    ch = torch.linalg.cholesky(h)
    inv_ch = torch.cholesky_inverse(ch)
    ch = torch.linalg.cholesky(inv_ch, upper=True)
    for idx in range(10):
        print(torch.max(torch.abs(ch[idx, idx:] * torch.sqrt(invh[0, 0]) - invh[0, :])))
        invh = smart(h, invh, 0)
        h = h[1:, 1:]
