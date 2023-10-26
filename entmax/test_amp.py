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
    Xs = [
        torch.randn(1000, dtype=torch.float32, device="cuda")
        for _ in range(5)
    ]


    @pytest.mark.parametrize("func", mappings)
    @pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
    def test_sum_one(func, dtype):
        _Xs = [_X.to(dtype) for _X in Xs]
        with torch.autocast(device_type="cuda", dtype=dtype):
            for _X in _Xs:
                scores = func(_X)
                prob_mass = scores.sum(-1)
                assert torch.allclose(prob_mass, torch.tensor([1.0], device="cuda"))
