import pytest
import torch
from functools import partial

from entmax import entmax15, sparsemax, entmax_bisect

torch.manual_seed(42)

def make_negatives(dtype, max_pow):
    negatives = []
    for i in range(2, max_pow + 1):
        negative = torch.randn(128, dtype=dtype, device="cuda") - 10 ** i
        negative[0] += 5
        negatives.append(negative)
    return negatives


if torch.cuda.is_available():

    mappings = [entmax15, sparsemax, partial(entmax_bisect, alpha=1.5), partial(entmax_bisect, alpha=2)]

    long_bf16 = [
        torch.randn(32000, dtype=torch.bfloat16, device="cuda")
        for _ in range(5)
    ]
    negatives_bf16 = make_negatives(torch.bfloat16, 7)

    long_fp16 = [
        torch.randn(32000, dtype=torch.float16, device="cuda")
        for _ in range(5)
    ]
    negatives_fp16 = make_negatives(torch.float16, 4)

    @pytest.mark.parametrize("Xs", (long_bf16, negatives_bf16, long_fp16, negatives_fp16))
    @pytest.mark.parametrize("func", mappings)
    def test_probs_close(Xs, func):
        dtype = Xs[0].dtype

        full_precision_probs = [func(X.to(torch.float32), dim=-1) for X in Xs]
        _Xs = [X.to(dtype) for X in Xs]
        with torch.autocast(device_type="cuda", dtype=dtype):
            for _X, fpp in zip(_Xs, full_precision_probs):
                probs = func(_X, dim=-1)
                assert torch.allclose(probs, fpp)
