import sys
from functools import partial
import time
import numpy as np
import torch

from entmax import (
    SparsemaxLoss,
    SparsemaxBisectLoss,
    Entmax15Loss,
    EntmaxBisectLoss,
)


def _bench(f):
    timings = []
    for _ in range(10):
        tic = time.perf_counter()
        f()
        torch.cuda.synchronize()
        toc = time.perf_counter()
        timings.append(toc - tic)
    return np.percentile(timings, [25, 50, 75])


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-size", type=int, dest="vocab_size", required=True
    )
    parser.add_argument("--top-k", dest="k", type=int, default=512)
    parser.add_argument("--n-iter", dest="n_iter", type=int, default=25)
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=1024
    )
    parser.add_argument("--device", dest="device", default="cpu")

    opt = parser.parse_args()
    print(opt)

    X = torch.randn(opt.batch_size, opt.vocab_size, device=opt.device)
    _, y = torch.max(torch.randn_like(X), dim=1)
    ix = y[0]

    args = dict(reduction="sum", ignore_index=ix)

    sp1 = partial(SparsemaxLoss(k=None, **args), X=X, target=y)
    sp2 = partial(SparsemaxLoss(k=opt.k, **args), X=X, target=y)
    sp3 = partial(SparsemaxBisectLoss(n_iter=opt.n_iter, **args), X=X, target=y)
    ts1 = partial(Entmax15Loss(k=None, **args), X=X, target=y)
    ts2 = partial(Entmax15Loss(k=opt.k, **args), X=X, target=y)
    ts3 = partial(
        EntmaxBisectLoss(alpha=1.5, n_iter=opt.n_iter, **args), X=X, target=y
    )

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    print("sparsemax topk", _bench(sp2))
    print("sparsemax full", _bench(sp1))
    print("sparsemax bis ", _bench(sp3))
    print("entmax15 topk", _bench(ts2))
    print("entmax15 full", _bench(ts1))
    print("entmax15 bis ", _bench(ts3))

    print(((sp1() - sp2()) ** 2).sum())
    print(((sp1() - sp3()) ** 2).sum())
    print(((ts1() - ts2()) ** 2).sum())
    print(((ts1() - ts3()) ** 2).sum())


if __name__ == "__main__":
    main()
