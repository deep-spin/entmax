import torch
import torch.nn as nn
from torch.autograd import Function

from entmax.activations import sparsemax, entmax15
from entmax.root_finding import entmax_bisect, sparsemax_bisect


class _GenericLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean"):
        assert reduction in ["elementwise_mean", "sum", "none", "mean"]
        if reduction == "elementwise_mean":
            reduction = "mean"
        self.reduction = reduction
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, X, target):
        if self.ignore_index is not None:
            num_samples = target.size(0)
            valid_positions = target != self.ignore_index
            target = target[valid_positions]
            X = X[valid_positions]

        loss = self.loss(X, target)

        if self.reduction == "none" and self.ignore_index is not None:
            nonzero_loss = loss
            loss = torch.zeros(num_samples, device=X.device)
            loss[valid_positions] = nonzero_loss
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class _GenericLossFunction(Function):
    @classmethod
    def forward(cls, ctx, X, target, alpha, proj_args):
        """
        X (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert X.shape[0] == target.shape[0]

        p_star = cls.project(X, alpha, **proj_args)
        loss = cls.omega(p_star, alpha)

        p_star.scatter_add_(1, target.unsqueeze(1), torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, X)
        ctx.save_for_backward(p_star)

        return loss

    @classmethod
    def backward(cls, ctx, grad_output):
        p_star, = ctx.saved_tensors
        grad = grad_output.unsqueeze(1) * p_star
        ret = (grad,)

        # pad with as many Nones as needed
        return ret + (None,) * (1 + cls.n_fwd_args)


class SparsemaxLossFunction(_GenericLossFunction):

    n_fwd_args = 1

    @classmethod
    def project(cls, X, alpha, k):
        return sparsemax(X, dim=-1, k=k)

    @classmethod
    def omega(cls, p_star, alpha):
        return (1 - (p_star ** 2).sum(dim=1)) / 2

    @classmethod
    def forward(cls, ctx, X, target, k=None):
        return super().forward(ctx, X, target, alpha=2, proj_args=dict(k=k))


class SparsemaxBisectLossFunction(_GenericLossFunction):

    n_fwd_args = 1

    @classmethod
    def project(cls, X, alpha, n_iter):
        return sparsemax_bisect(X, n_iter=n_iter)

    @classmethod
    def omega(cls, p_star, alpha):
        return (1 - (p_star ** 2).sum(dim=1)) / 2

    @classmethod
    def forward(cls, ctx, X, target, n_iter=50):
        return super().forward(
            ctx, X, target, alpha=2, proj_args=dict(n_iter=n_iter)
        )


class Entmax15LossFunction(_GenericLossFunction):

    n_fwd_args = 1

    @classmethod
    def project(cls, X, alpha, k=None):
        return entmax15(X, dim=-1, k=k)

    @classmethod
    def omega(cls, p_star, alpha):
        return (1 - (p_star * torch.sqrt(p_star)).sum(dim=1)) / 0.75

    @classmethod
    def forward(cls, ctx, X, target, k=None):
        return super().forward(ctx, X, target, alpha=1.5, proj_args=dict(k=k))


class EntmaxBisectLossFunction(_GenericLossFunction):

    n_fwd_args = 2

    @classmethod
    def project(cls, X, alpha, n_iter):
        return entmax_bisect(X, alpha=alpha, n_iter=n_iter, ensure_sum_one=True)

    @classmethod
    def omega(cls, p_star, alpha):
        return (1 - (p_star ** alpha).sum(dim=1)) / (alpha * (alpha - 1))

    @classmethod
    def forward(cls, ctx, X, target, alpha=1.5, n_iter=50):
        return super().forward(
            ctx, X, target, alpha, proj_args=dict(n_iter=n_iter)
        )


def sparsemax_loss(X, target, k=None):
    """sparsemax loss: sparse alternative to cross-entropy

    Computed using a partial sorting strategy.

    Parameters
    ----------
    X : torch.Tensor, shape=(n_samples, n_classes)
        The input 2D tensor of predicted scores

    target : torch.LongTensor, shape=(n_samples,)
        The ground truth labels, 0 <= target < n_classes.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    losses, torch.Tensor, shape=(n_samples,)
        The loss incurred at each sample.
    """
    return SparsemaxLossFunction.apply(X, target, k)


def sparsemax_bisect_loss(X, target, n_iter=50):
    """sparsemax loss: sparse alternative to cross-entropy

    Computed using bisection.

    Parameters
    ----------
    X : torch.Tensor, shape=(n_samples, n_classes)
        The input 2D tensor of predicted scores

    target : torch.LongTensor, shape=(n_samples,)
        The ground truth labels, 0 <= target < n_classes.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    Returns
    -------
    losses, torch.Tensor, shape=(n_samples,)
        The loss incurred at each sample.
    """
    return SparsemaxBisectLossFunction.apply(X, target, n_iter)


def entmax15_loss(X, target, k=None):
    """1.5-entmax loss: sparse alternative to cross-entropy

    Computed using a partial sorting strategy.

    Parameters
    ----------
    X : torch.Tensor, shape=(n_samples, n_classes)
        The input 2D tensor of predicted scores

    target : torch.LongTensor, shape=(n_samples,)
        The ground truth labels, 0 <= target < n_classes.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    losses, torch.Tensor, shape=(n_samples,)
        The loss incurred at each sample.
    """
    return Entmax15LossFunction.apply(X, target, k)


def entmax_bisect_loss(X, target, alpha=1.5, n_iter=50):
    """alpha-entmax loss: sparse alternative to cross-entropy

    Computed using bisection, supporting arbitrary alpha > 1.

    Parameters
    ----------
    X : torch.Tensor, shape=(n_samples, n_classes)
        The input 2D tensor of predicted scores

    target : torch.LongTensor, shape=(n_samples,)
        The ground truth labels, 0 <= target < n_classes.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use for each row of X. If scalar
        or python float, the same value is used for all rows. A value of
        alpha=2 corresponds to sparsemax, and alpha=1 corresponds to softmax
        (but computing it this way is likely unstable).

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    Returns
    -------
    losses, torch.Tensor, shape=(n_samples,)
        The loss incurred at each sample.
    """
    return EntmaxBisectLossFunction.apply(X, target, alpha, n_iter)


class SparsemaxBisectLoss(_GenericLoss):
    def __init__(
        self, n_iter=50, ignore_index=-100, reduction="elementwise_mean"
    ):
        self.n_iter = n_iter
        super(SparsemaxBisectLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return sparsemax_bisect_loss(X, target, self.n_iter)


class SparsemaxLoss(_GenericLoss):
    def __init__(self, k=None, ignore_index=-100, reduction="mean"):
        self.k = k
        super(SparsemaxLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return sparsemax_loss(X, target, self.k)


class EntmaxBisectLoss(_GenericLoss):
    def __init__(
        self,
        alpha=1.5,
        n_iter=50,
        ignore_index=-100,
        reduction="mean",
    ):
        self.alpha = alpha
        self.n_iter = n_iter
        super(EntmaxBisectLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return entmax_bisect_loss(X, target, self.alpha, self.n_iter)


class Entmax15Loss(_GenericLoss):
    def __init__(self, k=100, ignore_index=-100, reduction="mean"):
        self.k = k
        super(Entmax15Loss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return entmax15_loss(X, target, self.k)
