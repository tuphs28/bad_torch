import numpy as np

from badtorch.autograd import Tensor

def cross_entropy_loss(logits: Tensor, targets: Tensor, dim: int) -> Tensor:
    return - (targets * logits.logsumexp(dim=dim)).sum(dim=0)