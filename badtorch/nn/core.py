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
            xavier_normal_(self.linear)
        elif init == "xavier_uniform":
            xavier_uniform_(self.linear)
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

class Dropout(Module):

    def __init__(self, p: float = 0.05):
        self.p = p

    def __call__(self, input_tensor: Tensor) -> Tensor:
        mask = np.ones_like(input_tensor.data)
        mask[np.random.uniform(low=0, high=1, size=mask.shape) < self.p] = 0
        mask *= 1 / (1 - self.p)
        return Tensor(mask) * input_tensor
    
    def parameters(self) -> list[Tensor]:
        return []


class MLP(Module):

    def __init__(
        self,
        in_dims: list[int],
        out_dims: list[int],
        init: Literal["xavier_normal", "xavier_uniform"] = "xavier_normal",
        dropout: bool = True,
        dropout_p: float = 0.05
    ) -> None:
        
        assert len(in_dims) == len(out_dims), f"Need list of input and output dimensions to be the same, currently have {len(in_dims)} and {len(out_dims)}"

        self.training = True
        

        self.linear_layers = []
        for in_dim, out_dim in zip(in_dims, out_dims):
            self.linear_layers.append(
                Linear(in_dim=in_dim, out_dim=out_dim, bias=True, init=init)
            )
        
        self.dropout = None
        if dropout:
            self.dropout = Dropout(p=dropout_p)

    def __call__(self, input_tensor: Tensor) -> Tensor:
        x = input_tensor
        for idx, layer in enumerate(self.linear_layers[:-1]):
            x = layer(x) 
            x = x.relu()
            if self.training and self.dropout:
                x = self.dropout(x)
        logits = self.linear_layers[-1](x)
        return logits
    
    def parameters(self) -> list[Tensor]:
        parameters = []
        for layer in self.linear_layers:
            parameters += layer.parameters()
        return parameters
    
    def train_mode(self) -> None:
        self.training = True

    def eval_mode(self) -> None:
        self.training = False
    

        

        
