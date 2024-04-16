import torch


def upsampling_nearest2d(x: torch.Tensor, scale_factor: int):
    n, c, h, w = x.shape
    oh, ow = h*scale_factor, w*scale_factor
    x = x.view(n, c, h, 1, w, 1).expand(
        -1, -1, -1, scale_factor, -1, scale_factor).contiguous()
    x = x.view(n, c, oh, ow)
    return x


def cmp_with_cpu():
    n, c, h, w = 10, 3, 224, 224
    x = torch.randn(n, c, h, w)
    y = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
    my = unsampling_nearest2d(x, scale_factor=2)
    print(y.shape, my.shape)
    print((y - my).abs().max().item())


if __name__ == '__main__':
    cmp_with_cpu()

