from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from badtorch.autograd import Tensor
from badtorch.nn.init import xavier_normal_, xavier_uniform_

class Module(ABC):

    @abstractmethod
    def __call__(self, input_tensor: Tensor) -> Tensor:
        pass

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
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
        else:
            self.bias = None

    def __call__(self, input_tensor: Tensor) -> Tensor:
        output = input_tensor @ self.linear
        if self.bias is not None:
            output += self.bias
        return output

    def parameters(self) -> list[Tensor]:
        return [self.linear] + ([self.bias] if self.bias is not None else [])
    
    def __repr__(self) -> str:
        return f"Linear(in_dim={self.linear.shape[0]}, out_dim={self.linear.shape[1]}, bias={hasattr(self, 'bias')})"

