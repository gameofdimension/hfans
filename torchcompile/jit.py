import inspect
from typing import Dict, Optional

from torch._dynamo.bytecode_transformation import is_generator
from torch._dynamo.eval_frame import set_eval_frame  # type: ignore

from torch._dynamo.types import (
    CacheEntry,
    DynamoFrameType,
    FrameState,
    GuardedCode,
    GuardFn,
)


def full_function_name(frame):
    # Get the function name
    function_name = frame.f_code.co_name

    # Check if we are inside a class
    try:
        class_name = frame.f_locals["self"].__class__.__name__
        function_name = class_name + "." + function_name
    except KeyError:
        pass

    # Get the module name
    module = inspect.getmodule(frame)
    module_name = module.__name__ if module is not None else "empty module"

    return module_name + "." + function_name


class Always(GuardFn):
    def __init__(self, hit=True):
        self.hit = hit

    def __call__(self, f_locals: Dict) -> bool:
        print(f"guard called with {len(f_locals)} args.")
        return self.hit


class MyCallback:
    def __init__(self, skip: bool, hit: bool = False):
        self.count = 0
        self.skip = skip
        self.hit = hit

    def __call__(
        self,
        frame: DynamoFrameType,
        cache: Optional[CacheEntry],
        frame_state: FrameState,
    ):
        self.count += 1
        cache_len = 0
        while cache is not None:
            cache_len += 1
            cache = cache.next
        print(
            f"calling the {self.count}-th function: {full_function_name(frame)}")  # noqa
        print(f"the function has {cache_len} cache entries now.")

        if "cache_len" in frame_state:
            cache_len = frame_state["cache_len"]
            print(f"number of caches in the last call: {cache_len}")
        frame_state["cache_len"] = cache_len

        if self.skip or is_generator(frame.f_code):
            return None
        return GuardedCode(
            code=frame.f_code, check_fn=Always(hit=self.hit))  # type: ignore


def f(x):
    print(f"f called with {x}")
    return x + 1


def g(x):
    print(f"g called with {x}")
    return x + 2


def eval_frame_mwe(skip, hit=False):
    callback = MyCallback(skip=skip, hit=hit)
    prior = set_eval_frame(callback)

    x = 1
    x = f(x)
    x = g(x)
    x = f(x)
    x = g(x)
    x = f(x)

    set_eval_frame(prior)


def test_always_skip_callback():
    eval_frame_mwe(skip=True)


def test_always_hit_callback():
    eval_frame_mwe(skip=False, hit=True)


def test_always_miss_callback():
    eval_frame_mwe(skip=False, hit=False)


if __name__ == "__main__":
    # https://zhuanlan.zhihu.com/p/659050315
    # test_always_skip_callback()
    # test_always_hit_callback()
    test_always_miss_callback()
