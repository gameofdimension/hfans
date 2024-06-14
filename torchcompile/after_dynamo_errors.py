import torch

import torch._dynamo as dynamo

model = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(5)])
# toy compiler which fails if graph contains relu


def toy_compiler(gm: torch.fx.GraphModule, _):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            assert False

    return gm


def test_backend_error():
    y = torch.ones(200, 200)
    x = torch.ones(200, 200)
    z = x + y
    a = torch.relu(z)
    return model(a)


if __name__ == "__main__":
    compiled_test_backend_error = torch.compile(
        test_backend_error, backend=toy_compiler)
    compiled_test_backend_error()
