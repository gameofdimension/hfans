import torch
from torch import nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.list = nn.ModuleList([
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
        ])

    def forward(self, x):
        for i, layer in enumerate(self.list):
            print("Layer", i, "input dtype:", x.dtype,
                  "module type:", layer.__class__.__name__)
            x = layer(x)
            print("Layer", i, "output dtype:", x.dtype)
        return x


def main():
    device = 'cuda'
    model = Model().to(device)
    for param in model.named_parameters():
        print(param[0], param[1].dtype)
    data = torch.randn(4, 128, 64, 64, device=device)

    enabled = True
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enabled):
        out = model(data)

    print("Output dtype:", out.dtype)


if __name__ == "__main__":
    main()