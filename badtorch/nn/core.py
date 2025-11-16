from abc import ABC, abstractmethod
from typing import Literal, Any

import numpy as np

from badtorch.autograd import Tensor
from badtorch.nn.init import xavier_normal_, xavier_uniform_

class Module(ABC):

    def __init__(self):

        self.training = True

        self._modules = {}
        self._parameters = {}

    @abstractmethod
    def __call__(self, input_tensor: Tensor) -> Tensor:
        pass

    def parameters(self) -> list[Tensor]:
        parameters = []
        for module in self._modules.values():
            parameters.extend(module.parameters())
        parameters.extend(self._parameters.values())
        return parameters
    
    def param_count(self) -> int:
        n_params = 0
        for parameter in self.parameters():
            parameter_params = 1
            for dim_size in parameter.shape:
                parameter_params *= dim_size
            n_params += parameter_params
        return n_params

    def train_mode(self) -> None:
        for module in self._modules.values():
            module.train_mode()
        self.training = True

    def eval_mode(self) -> None:
        for module in self._modules.values():
            module.eval_mode()
        self.training = False

    def __setattr__(self, name: Any, value: Any) -> None:

        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        
        object.__setattr__(self, name, value)


class Linear(Module):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: Literal["xavier_normal", "xavier_uniform"] = "xavier_normal"
    ) -> None:
        
        super().__init__()

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
    
class Dropout(Module):

    def __init__(self, p: float = 0.05):
        super().__init__()
        self.p = p

    def __call__(self, input_tensor: Tensor) -> Tensor:
        if self.training:
            mask = np.ones_like(input_tensor.data)
            mask[np.random.uniform(low=0, high=1, size=mask.shape) < self.p] = 0
            mask *= 1 / (1 - self.p)
            output_tensor =  Tensor(mask) * input_tensor
        else:
            output_tensor = input_tensor
        return output_tensor
    

class ReLU(Module):

    def __call__(self, input_tensor: Tensor) -> Tensor:
        return input_tensor.relu()


class Sequential(Module):

    def __init__(self, module_list: list[Module]):
        super().__init__()
        self.module_list = module_list
        for idx, module in enumerate(self.module_list):
            self.__setattr__(f"sequential.component{idx}", module) 

    def __call__(self, input_tensor: Tensor) -> Tensor:
        x = input_tensor
        for module in self.module_list:
            x = module(x)
        return x
            

class MLP(Module):

    def __init__(
        self,
        in_dims: list[int],
        out_dims: list[int],
        init: Literal["xavier_normal", "xavier_uniform"] = "xavier_normal",
        dropout: bool = True,
        dropout_p: float = 0.05
    ) -> None:
        
        super().__init__()

        assert len(in_dims) == len(out_dims), f"Need list of input and output dimensions to be the same, currently have {len(in_dims)} and {len(out_dims)}"

        self.training = True
        

        network = []
        for idx, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            network.append(
                Linear(in_dim=in_dim, out_dim=out_dim, bias=True, init=init)
            )
            if dropout and idx != len(in_dims) - 1:
                network.append(Dropout(p=dropout_p))
            network.append(ReLU())
        self.network = Sequential(module_list=network)
        
        self.dropout = None

    def __call__(self, input_tensor: Tensor) -> Tensor:
        logits = self.network(input_tensor)
        return logits
  
    

        

        
