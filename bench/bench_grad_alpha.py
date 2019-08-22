import sys
from functools import partial
import time
import numpy as np
import torch
from torch.autograd import grad

from torch.nn.functional import softmax

from entmax import (
    sparsemax,
    entmax15,
    entmax_bisect
)


def bench(f_):
    timings_fwd = []
    timings_bck = []
    for _ in range(100):

        with f_ as f:
            tic = time.perf_counter()
            f.forward()
            torch.cuda.synchronize()
            toc = time.perf_counter()
            timings_fwd.append(toc - tic)

            tic = time.perf_counter()
            f.backward()
            torch.cuda.synchronize()
            toc = time.perf_counter()
            timings_bck.append(toc - tic)

    return (np.percentile(timings_fwd, [25, 50, 75]),
            np.percentile(timings_bck, [25, 50, 75]))


class MappingBencher(object):
    def __init__(self, mapping, X):
        self.mapping = mapping
        self.X_data = X

    def __enter__(self):
        self.X = self.X_data.clone().requires_grad_()
        self.dY = torch.randn_like(self.X)
        return self

    def forward(self):
        self.Y = self.mapping(self.X, dim=-1)

    def backward(self):
        grad(outputs=(self.Y,),
             inputs=(self.X,),
             grad_outputs=(self.Y))

    def __exit__(self, *args):
        try:
            del self.X
            del self.Y
        except AttributeError:
            pass


class EntmaxAlphaBencher(object):
    def __init__(self, X, n_iter=25):
        self.n_iter = n_iter
        self.X_data = X

    def __enter__(self):
        self.X = self.X_data.clone().requires_grad_()
        self.dY = torch.randn_like(self.X)
        self.alpha = 1.01 + torch.rand(self.X.shape[0], 1, device=self.X.device,
                                       requires_grad=True)
        return self

    def forward(self):
        self.Y = entmax_bisect(self.X, self.alpha, dim=-1, n_iter=self.n_iter)

    def backward(self):
        grad(outputs=(self.Y,),
             inputs=(self.X, self.alpha),
             grad_outputs=(self.Y))

    def __exit__(self, *args):
        try:
            del self.X
            del self.alpha
        except AttributeError:
            pass

        try:
            del self.Y
        except AttributeError:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", dest="dim", type=int, default=30)
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=64 * 8 * 30
    )
    parser.add_argument("--device", dest="device", default="cpu")

    opt = parser.parse_args()
    print(opt)

    X = torch.randn(opt.batch_size, opt.dim, device=opt.device)

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    print("softmax", bench(MappingBencher(softmax, X)))
    print("sparsemax", bench(MappingBencher(sparsemax, X)))
    print("entmax15", bench(MappingBencher(entmax15, X)))
    print("a-entmax 25iter", bench(EntmaxAlphaBencher(X, n_iter=25)))
    print("a-entmax 10iter", bench(EntmaxAlphaBencher(X, n_iter=10)))


if __name__ == "__main__":
    main()
