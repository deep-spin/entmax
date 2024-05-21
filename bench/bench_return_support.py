import argparse

import matplotlib.pyplot as plt
import torch

from entmax import entmax15

from bench_topk import bench_topk


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

    results["no_support"] = bench_topk(args.k, num_threads, batch, n=args.n, ntimeit=args.ntimeit, logmin=-1.5, logmax=2)
    results["support"] = bench_topk(args.k, num_threads, batch, n=args.n, ntimeit=args.ntimeit, logmin=-1.5, logmax=2, return_support_size=True)


    for k, v in results.items():
        print(k)
        plt.plot(v[0], v[1], label=k)
    # plt.xticks(args.k)
    plt.legend()
    plt.savefig(args.out, format="pdf", bbox_inches="tight", dpi=2000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--ntimeit", type=int, default=20)
    parser.add_argument("--v", type=int, default=32000)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", default="fig.pdf")
    args = parser.parse_args()
    main(args)
