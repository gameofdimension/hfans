import torch


def avgpool2d(x, kernel_size):
    n, c, h, w = x.shape
    oh, ow = h // kernel_size, w // kernel_size

    x = x.view(n, c, oh, kernel_size, ow, kernel_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    y = x.view(n, c, oh, ow, -1).mean(dim=-1)
    return y


def main():
    n, c, h, w = 10, 3, 224, 224
    x = torch.randn(n, c, h, w)
    y = torch.nn.functional.avg_pool2d(x, kernel_size=2)

    my = avgpool2d(x, kernel_size=2)
    print(y.shape, my.shape)
    print((y - my).abs().max().item())


if __name__ == '__main__':
    main()
