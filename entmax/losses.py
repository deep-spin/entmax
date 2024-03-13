import torch
import torch.nn as nn
from torch.autograd import Function

from entmax.activations import sparsemax, entmax15
from entmax.root_finding import entmax_bisect, sparsemax_bisect, normmax_bisect


class _GenericLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean", return_support=False):
        assert reduction in ["elementwise_mean", "sum", "none"]
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.return_support = return_support
        super(_GenericLoss, self).__init__()

    def forward(self, X, target):

        loss = self.loss(X, target)
        support = None
        if isinstance(loss, tuple):
            # second element is the support
            loss, support = loss

        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "elementwise_mean":
            loss = loss.sum() / size

        if self.return_support:
            return loss, support
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
        support_size = None
        if isinstance(p_star, tuple):
            p_star, support_size = p_star

        loss = cls.omega(p_star, alpha)

        p_star.scatter_add_(1, target.unsqueeze(1), torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, X)
        ctx.save_for_backward(p_star)

        return loss, support_size

    @classmethod
    def backward(cls, ctx, grad_output, supp):
        p_star, = ctx.saved_tensors
        grad = grad_output.unsqueeze(1) * p_star
        ret = (grad,)

        # pad with as many Nones as needed
        return ret + (None,) * (1 + cls.n_fwd_args)


class SparsemaxLossFunction(_GenericLossFunction):

    n_fwd_args = 2

    @classmethod
    def project(cls, X, alpha, k, return_support=False):
        return sparsemax(X, dim=-1, k=k, return_support=return_support)

    @classmethod
    def omega(cls, p_star, alpha):
        return (1 - (p_star ** 2).sum(dim=1)) / 2

    @classmethod
    def forward(cls, ctx, X, target, k=None, return_support=False):
        return super().forward(
            ctx,
            X,
            target,
            alpha=2,
            proj_args=dict(k=k, return_support=return_support)
        )


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

    n_fwd_args = 2

    @classmethod
    def project(cls, X, alpha, k=None, return_support=False):
        return entmax15(X, dim=-1, k=k, return_support=return_support)

    @classmethod
    def omega(cls, p_star, alpha):
        return (1 - (p_star * torch.sqrt(p_star)).sum(dim=1)) / 0.75

    @classmethod
    def forward(cls, ctx, X, target, k=None, return_support=False):
        return super().forward(
            ctx,
            X,
            target,
            alpha=1.5,
            proj_args=dict(k=k, return_support=return_support)
        )


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


class NormmaxBisectLossFunction(_GenericLossFunction):

    n_fwd_args = 2

    @classmethod
    def project(cls, X, alpha, n_iter):
        return normmax_bisect(X, alpha=alpha, n_iter=n_iter)

    @classmethod
    def omega(cls, p_star, alpha):
        return 1 - (p_star ** alpha).sum(dim=1) ** (1 / alpha)

    @classmethod
    def forward(cls, ctx, X, target, alpha=2, n_iter=50):
        return super().forward(
            ctx, X, target, alpha, proj_args=dict(n_iter=n_iter)
        )


def sparsemax_loss(X, target, k=None, return_support=False):
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
    return SparsemaxLossFunction.apply(X, target, k, return_support)


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


def entmax15_loss(X, target, k=None, return_support=False):
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
    return Entmax15LossFunction.apply(X, target, k, return_support)


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
        alpha=2 corresponds to sparsemax, and alpha=1 would in theory recover
        softmax. For numeric reasons, this algorithm does not work with `alpha=1`:
        if you want softmax, we recommend `torch.nn.softmax`

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    Returns
    -------
    losses, torch.Tensor, shape=(n_samples,)
        The loss incurred at each sample.
    """
    return EntmaxBisectLossFunction.apply(X, target, alpha, n_iter)


def normmax_bisect_loss(X, target, alpha=2, n_iter=50):
    """alpha-normmax loss: another sparse alternative to cross-entropy

    Computed using bisection, supporting arbitrary alpha > 1.

    Parameters
    ----------
    X : torch.Tensor, shape=(n_samples, n_classes)
        The input 2D tensor of predicted scores

    target : torch.LongTensor, shape=(n_samples,)
        The ground truth labels, 0 <= target < n_classes.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use for each row of X. If scalar
        or python float, the same value is used for all rows.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    Returns
    -------
    losses, torch.Tensor, shape=(n_samples,)
        The loss incurred at each sample.
    """
    return NormmaxBisectLossFunction.apply(X, target, alpha, n_iter)


class SparsemaxLoss(_GenericLoss):
    def __init__(
        self,
        k=None,
        ignore_index=-100,
        reduction="elementwise_mean",
        return_support=False
    ):
        self.k = k
        super(SparsemaxLoss, self).__init__(
            ignore_index, reduction, return_support
        )

    def loss(self, X, target):
        return sparsemax_loss(X, target, self.k, self.return_support)


class Entmax15Loss(_GenericLoss):
    def __init__(
        self,
        k=100,
        ignore_index=-100,
        reduction="elementwise_mean",
        return_support=False
    ):
        self.k = k
        super(Entmax15Loss, self).__init__(
            ignore_index, reduction, return_support
        )

    def loss(self, X, target):
        return entmax15_loss(X, target, self.k, self.return_support)


class SparsemaxBisectLoss(_GenericLoss):
    def __init__(
        self,
        n_iter=50,
        ignore_index=-100,
        reduction="elementwise_mean"
    ):
        self.n_iter = n_iter
        super(SparsemaxBisectLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return sparsemax_bisect_loss(X, target, self.n_iter)


class EntmaxBisectLoss(_GenericLoss):
    def __init__(
        self,
        alpha=1.5,
        n_iter=50,
        ignore_index=-100,
        reduction="elementwise_mean"
    ):
        self.alpha = alpha
        self.n_iter = n_iter
        super(EntmaxBisectLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return entmax_bisect_loss(X, target, self.alpha, self.n_iter)


class NormmaxBisectLoss(_GenericLoss):
    def __init__(
        self,
        alpha=2,
        n_iter=50,
        ignore_index=-100,
        reduction="elementwise_mean"
    ):
        self.alpha = alpha
        self.n_iter = n_iter
        super(NormmaxBisectLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return normmax_bisect_loss(X, target, self.alpha, self.n_iter)
