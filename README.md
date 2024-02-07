[![Build Status](https://dev.azure.com/zephyr14/entmax/_apis/build/status/deep-spin.entmax?branchName=master)](https://dev.azure.com/zephyr14/entmax/_build/latest?definitionId=1&branchName=master)

[![PyPI version](https://badge.fury.io/py/entmax.svg)](https://badge.fury.io/py/entmax)

# entmax

<img src="entmax.png" />

--------------------------------------------------------------------------------

This package provides a pytorch implementation of entmax and entmax losses:
a sparse family of probability mappings and corresponding loss functions,
generalizing softmax / cross-entropy.

*Features:*
  - Exact partial-sort algorithms for 1.5-entmax and 2-entmax (sparsemax).
  - A bisection-based algorithm for generic alpha-entmax.
  - Gradients w.r.t. alpha for adaptive, learned sparsity!
  - Other sparse transformations: alpha-normmax and k-subsets budget (handled through bisection-based algorithms).

*Requirements:* python 3, pytorch >= 1.3 (and pytest for unit tests)

## Example

```python
In [1]: import torch

In [2]: from torch.nn.functional import softmax

In [2]: from entmax import sparsemax, entmax15, entmax_bisect, normmax_bisect, budget_bisect

In [4]: x = torch.tensor([-2, 0, 0.5])

In [5]: softmax(x, dim=0)
Out[5]: tensor([0.0486, 0.3592, 0.5922])

In [6]: sparsemax(x, dim=0)
Out[6]: tensor([0.0000, 0.2500, 0.7500])

In [7]: entmax15(x, dim=0)
Out[7]: tensor([0.0000, 0.3260, 0.6740])

In [8]: normmax_bisect(x, alpha=2, dim=0)
Out[8]: tensor([0.0000, 0.3110, 0.6890])

In [9]: normmax_bisect(x, alpha=1000, dim=0)
Out[9]: tensor([0.0000, 0.4997, 0.5003])

In [10]: budget_bisect(x, budget=2, dim=0)
Out[10]: tensor([0.0000, 1.0000, 1.0000])
```

Gradients w.r.t. alpha (continued):

```python
In [1]: from torch.autograd import grad

In [2]: x = torch.tensor([[-1, 0, 0.5], [1, 2, 3.5]])

In [3]: alpha = torch.tensor(1.33, requires_grad=True)

In [4]: p = entmax_bisect(x, alpha)

In [5]: p
Out[5]:
tensor([[0.0460, 0.3276, 0.6264],
        [0.0026, 0.1012, 0.8963]], grad_fn=<EntmaxBisectFunctionBackward>)

In [6]: grad(p[0, 0], alpha)
Out[6]: (tensor(-0.2562),)
```

## Installation

```
pip install entmax
```

## Citations

[Sparse Sequence-to-Sequence Models](https://www.aclweb.org/anthology/P19-1146)

```
@inproceedings{entmax,
  author    = {Peters, Ben and Niculae, Vlad and Martins, Andr{\'e} FT},
  title     = {Sparse Sequence-to-Sequence Models},
  booktitle = {Proc. ACL},
  year      = {2019},
  url       = {https://www.aclweb.org/anthology/P19-1146}
}
```

[Adaptively Sparse Transformers](https://arxiv.org/pdf/1909.00015.pdf)

```
@inproceedings{correia19adaptively,
  author    = {Correia, Gon\c{c}alo M and Niculae, Vlad and Martins, Andr{\'e} FT},
  title     = {Adaptively Sparse Transformers},
  booktitle = {Proc. EMNLP-IJCNLP (to appear)},
  year      = {2019},
}
```

Further reading:

  - Blondel, Martins, and Niculae, 2019. [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324).
  - Martins and Astudillo, 2016. [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
  - Peters and Martins, 2019 [IT-IST at the SIGMORPHON 2019 Shared Task: Sparse Two-headed Models for Inflection](https://www.aclweb.org/anthology/W19-4207).
