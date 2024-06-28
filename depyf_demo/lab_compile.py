import torch
from torch import nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, 128)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2)

        self.norm2 = nn.GroupNorm(32, 256)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, x):
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.pool(x)
        print("will trigger graph break, x.shape:", x.shape)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


def main():
    device = 'cuda'
    model = Model().to(device)
    model = torch.compile(model=model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    enabled = True

    step = 0
    while True:
        data = torch.randn(64, 128, 64, 64, device=device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enabled):  # type: ignore # noqa
            out = model(data)
            print("return type", type(out))
            if isinstance(out, list):
                out = out[0]
            loss = (out*out).mean()
            print("Loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step >= 10:
            break


def will_return_loss_list():
    import depyf
    with depyf.prepare_debug("depyf_debug_dir"):
        main()


def will_return_loss_tensor():
    main()


if __name__ == '__main__':
    will_return_loss_list()
    # will_return_loss_tensor()
