import torch


@torch.compile
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def main():
    for _ in range(100):
        toy_example(
            torch.randn(10, device='cuda'), torch.randn(10, device='cuda'))


def run_compile_debug():
    main()


def run_depyf():
    import depyf
    with depyf.prepare_debug("depyf_debug_dir"):
        main()


if __name__ == "__main__":
    # https://depyf.readthedocs.io/en/latest/faq.html#what-is-the-difference-between-depyf-and-torch-compile-debug
    run_depyf()
