from abc import ABC, abstractmethod

import numpy as np

from badtorch.autograd import Tensor

class Optimiser(ABC):

    def __init__(self, parameters: list[Tensor], lr: float = 1e-3) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.grad)


class SGD(Optimiser):

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data = parameter.data - self.lr * parameter.grad