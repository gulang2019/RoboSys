import time 
from contextlib import contextmanager 

def _time(f: callable):
    import torch 
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10
    return elapsed


@contextmanager
def timer(scope: str):
    try:
        start = time.perf_counter()
        yield 
    finally:
        elapsed = time.perf_counter() - start
        print('%s takes %.2f ms' % (scope, elapsed * 1000))
