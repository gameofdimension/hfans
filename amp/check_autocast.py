import torch


def check_autocast():
    a = torch.randn(100, 200).cuda()
    b = torch.randn(200, 100).cuda()
    c = a@b
    assert c.dtype == torch.float32

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        c = a@b
        assert c.dtype == torch.bfloat16

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        c = a@b
        assert c.dtype == torch.bfloat16


if __name__ == "__main__":
    check_autocast()

