import logging
import torch._inductor.config as icfg
import torch._functorch.config as fcfg
import torch._dynamo.config as dcfg
import torch
import torch.nn as nn
from triton.testing import do_bench

# torch._logging.set_logs(
#     dynamo=logging.DEBUG, aot=logging.DEBUG, inductor=logging.DEBUG)
# dcfg.log_level = logging.DEBUG
# dcfg.print_graph_breaks = True
# dcfg.output_code = True

icfg.debug = True
icfg.trace.enabled = True


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.fc1(x).relu() ** 2
        return self.fc2(x).relu() ** 2


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    mod = MLP().cuda()
    opt_mod = torch.compile(mod)
    x = torch.randn(1024, 1024, device="cuda")

    base_time = do_bench(
        lambda: mod(x).sum().backward(), grad_to_none=mod.parameters()
    )
    opt_time = do_bench(
        lambda: opt_mod(x).sum().backward(), grad_to_none=mod.parameters()
    )
    print(f"speedup: {base_time / opt_time:.2f}")
