import numpy as np

from badtorch.autograd import Tensor

def cross_entropy_loss(logits: Tensor, targets: Tensor, dim: int) -> Tensor:
    log_probs = logits - logits.logsumexp(dim=dim)
    return - (targets * log_probs).sum(dim=0).sum(dim=1)