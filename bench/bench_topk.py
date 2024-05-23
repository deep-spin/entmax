import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.benchmark as benchmark

from entmax import entmax15


def bench_topk(k, num_threads, batch, logmin=-3, logmax=3, n=20, ntimeit=10, return_support_size=False):
    supports = []
    runtimes = []
    for b in np.logspace(logmin, logmax, n):
        inp = batch * b

        p, supp_raw = entmax15(inp, return_support_size=True, k=k)
        supp = supp_raw.float().mean().cpu().item()
        print(b, supp, k)
        if return_support_size:
            stmt = 'entmax15(x, k=k, return_support_size=True)'
        else:
            stmt = 'entmax15(x, k=k)'
        t0 = benchmark.Timer(
            stmt=stmt,
            setup='from __main__ import entmax15',
            num_threads=num_threads,
            globals={'x': inp, 'k': k}
        )

        supports.append(supp)
        measurement = t0.timeit(ntimeit)
        runtimes.append(measurement.mean)
    return supports, runtimes


# the question: does support size explain all the variance, or does entmax
# also have high variance even with a consistent support size?
def main(args):
    # we want to understand how the runtime develops as a function of sparsity.
    # do we want to think about the loss too?

    num_threads = torch.get_num_threads()
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    batch = torch.randn(args.batch, args.v, device=device)

    results = dict()
    # results["full"] = bench(None, num_threads, batch, n=args.n, ntimeit=20, logmin=-1.5, logmax=2)
    for k in args.k:
        results[k] = bench_topk(k, num_threads, batch, n=args.n, ntimeit=args.ntimeit, logmin=-1.5, logmax=2)

    # the action is all happening at the smaller support sizes. I should focus on them.

    for k, v in results.items():
        print(k)
        plt.plot(v[0], v[1], label=k)
    plt.xticks(args.k)
    plt.legend()
    plt.savefig(args.out, format="pdf", bbox_inches="tight", dpi=2000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--ntimeit", type=int, default=20)
    parser.add_argument("--v", type=int, default=32000)
    parser.add_argument("--k", nargs="+", type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", default="fig.pdf")
    args = parser.parse_args()
    main(args)
