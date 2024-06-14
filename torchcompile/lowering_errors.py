import torch


model = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(5)])


def test_backend_error():

    y = torch.ones(200, 200)
    x = torch.ones(200, 200)
    z = x + y
    a = torch.ops.aten._foobar(z)  # dummy function which errors # type: ignore
    return model(a)


if __name__ == "__main__":
    """
    https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#diagnosing-torchinductor-errors
    已无法重现
    """
    compiled_test_backend_error = torch.compile(
        test_backend_error, backend="inductor")
    compiled_test_backend_error()
