__version__ = "1.3"

from entmax.activations import sparsemax, entmax15, Sparsemax, Entmax15
from entmax.root_finding import (
    sparsemax_bisect,
    entmax_bisect,
    normmax_bisect,
    budget_bisect,
    SparsemaxBisect,
    EntmaxBisect,
    NormmaxBisect,
    BudgetBisect
)
from entmax.losses import (
    sparsemax_loss,
    entmax15_loss,
    sparsemax_bisect_loss,
    entmax_bisect_loss,
    normmax_bisect_loss,
    SparsemaxLoss,
    SparsemaxBisectLoss,
    Entmax15Loss,
    EntmaxBisectLoss,
    NormmaxBisectLoss
)
