import torch


def test_assertion_error():
    y = torch.ones(200, 200)
    z = {y: 5}
    return z


if __name__ == "__main__":
    """
    backend="eager" 也会报错，说明是 dynamo 的问题，也就是 graph capture 不 work
    """
    # https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#torchdynamo-errors
    compiled_test_assertion_error = torch.compile(
        test_assertion_error, backend="eager")

    # compiled_test_assertion_error fail while test_assertion_error pass
    # test_assertion_error()
    compiled_test_assertion_error()
