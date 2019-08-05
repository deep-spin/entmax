import pytest
import torch
from torch.autograd import gradcheck


from entmax.activations import (
    _threshold_and_support,
    _entmax_threshold_and_support,
    Sparsemax,
    Entmax15,
)

from entmax.losses import SparsemaxLoss, Entmax15Loss


@pytest.mark.parametrize('dim', (0, 1, 2))
@pytest.mark.parametrize('Map', (Sparsemax, Entmax15))
def test_mapping(dim, Map):
    f = Map(dim=dim, k=3)

    for _ in range(10):
        x = torch.randn(5, 6, 7, requires_grad=True, dtype=torch.float64)
        gradcheck(f, (x,))


@pytest.mark.parametrize('dim', (0, 1, 2))
@pytest.mark.parametrize('coef', (0.00001, 0.5, 10000))
def test_entmax_topk(dim, coef):
    x = coef * torch.randn(10, 11, 12)
    tau1, supp1 = _entmax_threshold_and_support(x, dim=dim)
    tau2, supp2 = _entmax_threshold_and_support(x, dim=dim, k=5)

    assert torch.all(tau1 == tau2)
    assert torch.all(supp1 == supp2)


@pytest.mark.parametrize('dim', (0, 1, 2))
@pytest.mark.parametrize('coef', (0.00001, 0.5, 10000))
@pytest.mark.parametrize('k', (5, 30))
def test_sparsemax_topk(dim, coef, k):

    x = coef * torch.randn(10, 11, 12)
    tau1, supp1 = _threshold_and_support(x, dim=dim)
    tau2, supp2 = _threshold_and_support(x, dim=dim, k=k)

    assert torch.all(tau1 == tau2)
    assert torch.all(supp1 == supp2)


def _bench(f):
    timings = []
    for _ in range(10):
        tic = time.perf_counter()
        f()
        torch.cuda.synchronize()
        toc = time.perf_counter()
        timings.append(toc - tic)
    return np.percentile(timings, [25, 50, 75])


def check_speed():
    import sys
    # device = 'cpu'
    # device = 'cuda'

    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'

    print("vocab={} k={} n_iter={} device={}".format(vocab_size, k, n_iter, device))

    x = torch.randn(1024, vocab_size, device=device)
    _, y = torch.max(torch.randn_like(x), dim=1)
    ix = y[0]

    args = dict(reduction='sum', ignore_index=ix)

    from entmax.losses import SparsemaxBisectLoss, EntmaxBisectLoss

    sp1 = partial(SparsemaxLoss(**args), input=x, target=y)
    sp2 = partial(SparsemaxLoss(k=k, **args), input=x, target=y)
    sp3 = partial(SparsemaxBisectLoss(n_iter=n_iter, **args), input=x, target=y)
    ts1 = partial(Entmax15Loss(**args), input=x, target=y)
    ts2 = partial(Entmax15Loss(k=k, **args), input=x, target=y)
    ts3 = partial(EntmaxBisectLoss(alpha=1.5, n_iter=n_iter, **args), input=x, target=y)

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


if __name__ == '__main__':
     import numpy as np
     from functools import partial
     import time
     check_speed()
