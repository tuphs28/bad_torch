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


class Adam(Optimiser):

    def __init__(self, parameters: list, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__(parameters=parameters, lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m1s = []
        self.m2s = []
        for parameter in self.parameters:
            self.m1s.append(np.zeros_like(parameter.grad))
            self.m2s.append(np.zeros_like(parameter.grad))

    def step(self) -> None:
        self.t += 1

        for parameter_idx, parameter in enumerate(self.parameters):

            self.m1s[parameter_idx] = self.beta1 * self.m1s[parameter_idx] + (1-self.beta1) * parameter.grad
            self.m2s[parameter_idx] = self.beta2 * self.m2s[parameter_idx] + (1-self.beta2) * parameter.grad ** 2

            m1_hat = self.m1s[parameter_idx] / (1 - self.beta1 ** self.t)
            m2_hat = self.m2s[parameter_idx] / (1 - self.beta2 ** self.t)

            parameter.data = parameter.data -  self.lr * m1_hat / (np.sqrt(m2_hat) + self.eps)
