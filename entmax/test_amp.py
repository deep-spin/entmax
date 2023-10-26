import pytest
import torch
from functools import partial

from entmax import entmax15, sparsemax, entmax_bisect


# These tests only work on cuda, so the first test will fail if you do not have it


def test_cuda_available():
    assert torch.cuda.is_available()


if torch.cuda.is_available():

    mappings = [entmax15, sparsemax, partial(entmax_bisect, alpha=1.5), partial(entmax_bisect, alpha=2)]

    # make data
    long_vecs = [
        torch.randn(32000, dtype=torch.float32, device="cuda")
        for _ in range(5)
    ]

    negatives = []
    for i in range(2, 6):
        negative = torch.randn(128, dtype=torch.float32, device="cuda") - 10 ** i
        negative[0] += 5
        negatives.append(negative)

    @pytest.mark.parametrize("Xs", (long_vecs, negatives))
    @pytest.mark.parametrize("func", mappings)
    @pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
    def test_sum_one(Xs, func, dtype):
        _Xs = [X.to(dtype) for X in Xs]
        with torch.autocast(device_type="cuda", dtype=dtype):
            for _X in _Xs:
                scores = func(_X, dim=-1)
                prob_mass = scores.sum(-1)
                assert torch.allclose(prob_mass, torch.tensor([1.0], device="cuda"))

    @pytest.mark.parametrize("Xs", (long_vecs, negatives))
    @pytest.mark.parametrize("func", mappings)
    @pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
    def test_probs_close(Xs, func, dtype):
        full_precision_probs = [func(X.to(dtype).to(torch.float32), dim=-1) for X in Xs]
        _Xs = [X.to(dtype) for X in Xs]
        with torch.autocast(device_type="cuda", dtype=dtype):
            for _X, fpp in zip(_Xs, full_precision_probs):
                probs = func(_X, dim=-1)
                assert torch.allclose(probs, fpp)
