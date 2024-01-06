import torch

from typing import List


def fn(x):
    a = torch.cos(x)
    b = torch.sin(a)
    return b


def demo_basic():
    # run with `TORCH_COMPILE_DEBUG=1``
    new_fn = torch.compile(fn, backend="inductor")
    input_tensor = torch.randn(10000).to(device="cuda:0")
    a = new_fn(input_tensor)
    return a


def custom_backend(
        gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


def bar(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b


def demo_display_graph():
    opt_bar = torch.compile(bar, backend=custom_backend)
    inp1 = torch.randn(10)
    inp2 = torch.randn(10)
    opt_bar(inp1, inp2)
    opt_bar(inp1, -inp2)


def demo_explain():
    # Reset since we are using a different backend.
    torch._dynamo.reset()
    explain_output = torch._dynamo.explain(
        bar, torch.randn(10), torch.randn(10))
    # print(explain_output.break_reasons)
    # print(explain_output.out_guards)
    print(explain_output)


def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.Dropout(p=0.2))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()


def demo_backward():
    model = make_model(128, 64, 3)
    model = torch.compile(model)

    data = torch.randn(16, 128, device='cuda')
    out = model(data)
    loss = out.sum()
    loss.backward()


#  TORCH_COMPILE_DEBUG=1 python demo_compile.py
if __name__ == '__main__':
    # demo_basic()
    # demo_display_graph()
    # demo_explain()
    demo_backward()
