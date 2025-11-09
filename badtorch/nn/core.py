from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from badtorch.autograd import Tensor
from badtorch.nn.init import xavier_normal_, xavier_uniform_

class Module:

    @abstractmethod
    def __call__(self, input_tensor: Tensor) -> Tensor:
        pass

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        pass


class Linear(Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: Literal["xavier_normal", "xavier_uniform"] = "xavier_normal"
    ) -> None:
        
        
        self.linear = Tensor(data=np.ones(shape=(in_dim, out_dim)), requires_grad=True)
        if init == "xavier_normal":
            self.linear = xavier_normal_(self.linear)
        elif init == "xavier_uniform":
            self.linear = xavier_uniform_(self.linear)
        else:
            raise ValueError(f"Unrecognised init type: {init}")
        
        if bias:
            self.bias = Tensor(np.zeros(shape=(out_dim,)), requires_grad=True)

    def __call__(self, input_tensor: Tensor) -> Tensor:
        return input_tensor @ self.linear + self.bias
    
    def parameters(self) -> list[Tensor]:
        return [self.linear, self.bias]

