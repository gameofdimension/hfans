import torch


def foo(x):
    s = torch.sin(x)
    c = torch.cos(x)
    return s + c


def gbreak(x):
    r = torch.sin(x) + torch.cos(x)
    if r.sum() < 0:
        return r + torch.tan(x)
    else:
        return r - torch.tan(x)


def demo_gbreak():
    compiled_gbreak = torch.compile(
        gbreak, options={"trace.enabled": True, "trace.graph_diagram": True})
    compiled_gbreak(torch.tensor(range(10)))


def demo_break_reason():
    explained = torch._dynamo.explain(gbreak, torch.tensor(range(10)))
    print(explained.break_reasons)
    for g in explained.graphs:
        g.graph.print_tabular()
        print()


def main():
    compiled_foo = torch.compile(
        foo, options={"trace.enabled": True, "trace.graph_diagram": True})
    compiled_foo(torch.tensor(range(10)))


if __name__ == "__main__":
    # main()
    # demo_gbreak()
    demo_break_reason()
