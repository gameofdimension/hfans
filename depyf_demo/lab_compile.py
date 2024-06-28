import torch
from torch import nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(128)
        self.act1 = nn.Sigmoid()
        self.mm1 = nn.Linear(128, 256)

        self.norm2 = nn.LayerNorm(256)
        self.act2 = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.mm2 = nn.Linear(256, 256)

    def forward(self, x):
        x = self.norm1(x)
        x = self.act1(x)
        x = self.mm1(x)
        x = 42*x
        print("will trigger graph break, x.shape:", x.shape)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.mm2(x)
        return x


def main():
    device = 'cuda'
    model = Model().to(device)
    model = torch.compile(model=model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    enabled = True

    step = 0
    while True:
        data = torch.randn(64, 128, device=device)
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
