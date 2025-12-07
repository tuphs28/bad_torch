from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np

from .core import Tensor

class Function(ABC):

    def __init__(self, context: dict[str, Tensor], parents: list[Tensor]) -> None:
        self.context = context
        self.parents = parents

    @classmethod
    def apply(cls, *input_tensors: Union[int, float, Tensor, np.ndarray], **kwargs) -> Tensor:
        
        input_tensors = tuple((t if isinstance(t, Tensor) else Tensor(t, requires_grad=False)) for t in input_tensors)

        result_requires_grad = any(t.requires_grad for t in input_tensors)
        result, context = cls.forward(*input_tensors, result_requires_grad=result_requires_grad, **kwargs)

        if not result_requires_grad:
            return result
        
        parents = [t for t in input_tensors]
        fnc = cls(context=context, parents=parents)
        result._creator = fnc
        for i, t in enumerate(input_tensors):
            if not t.requires_grad:
                continue
            vjp = fnc.__getattribute__(f"vjp{i}")
            result._backward_fncs.append(vjp)
        return result

    @staticmethod
    @abstractmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        pass

    
    @staticmethod
    def _unbroadcast_gradient(grad: np.ndarray, data_shape: tuple[int]) -> np.ndarray:

        ndims_added = len(grad.shape) - len(data_shape)
        if ndims_added > 0:
            grad = np.sum(grad, axis=tuple(range(ndims_added)))

        for dim_idx, (dim_grad, dim_data) in enumerate(zip(grad.shape, data_shape)):
            if dim_grad > 1 and dim_data == 1:
                grad = np.sum(grad, axis=dim_idx, keepdims=True)
        
        return grad
    

class BinaryFunction(Function):
    @abstractmethod
    def vjp0(self) -> None:
        pass
    @abstractmethod
    def vjp1(self) -> None:
        pass


class UnaryFunction(Function):
    @abstractmethod
    def vjp0(self) -> None:
        pass


class Add(BinaryFunction):
    
    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        input_0, input_1 = input_tensors
        data = input_0.data + input_1.data
        result = Tensor(data=data, requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "input_1": input_1,
            "result": result
        }
        return result, context
    
    def vjp0(self) -> None:
        grad = self.context["result"].grad
        grad = self._unbroadcast_gradient(grad, self.context["input_0"].shape)
        self.context["input_0"].grad += grad

    def vjp1(self) -> None:
        grad = self.context["result"].grad
        grad = self._unbroadcast_gradient(grad, self.context["input_1"].shape)
        self.context["input_1"].grad += grad


class Mul(BinaryFunction):

    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        input_0, input_1 = input_tensors
        data = input_0.data * input_1.data
        result = Tensor(data=data, requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "input_1": input_1,
            "result": result
        }
        return result, context
    
    def vjp0(self):
        grad = self.context["result"].grad * self.context["input_1"].data
        grad = self._unbroadcast_gradient(grad, self.context["input_0"].shape)
        self.context["input_0"].grad += grad

    def vjp1(self):
        grad = self.context["result"].grad * self.context["input_0"].data
        grad = self._unbroadcast_gradient(grad, self.context["input_1"].shape)
        self.context["input_1"].grad += grad


class MatMul(BinaryFunction):

    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        input_0, input_1 = input_tensors
        data = input_0.data @ input_1.data
        result = Tensor(data=data, requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "input_1": input_1,
            "result": result
        }
        return result, context
    
    def vjp0(self):
        input_0, input_1, result = self.context.values()
        other_data = input_1.data if len(input_1.data.shape) > 1 else input_1.data[:,None]
        result_grad = result.grad if len(result.grad.shape) > 1 else result.grad[:,None]
        grad = result_grad @ other_data.swapaxes(-1, -2)
        grad = grad if len(input_0.shape) > 1 else grad[:,0]
        self.context["input_0"].grad += grad

    def vjp1(self):
        input_0, input_1, result = self.context.values()
        self_data = input_0.data if len(input_0.data.shape) > 1 else input_0.data[:,None]
        result_grad = result.grad if len(result.grad.shape) > 1 else result.grad[:,None]
        grad = self_data.swapaxes(-1, -2) @ result_grad
        grad = grad if len(input_1.data.shape) > 1 else grad[:,0]
        input_1.grad += grad
        

class ReLU(UnaryFunction):

    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        input_0, = input_tensors
        mask = Tensor(input_0.data <= 0.0)
        data_masked = input_0.data.copy()
        data_masked[mask.data] = 0
        result = Tensor(data=data_masked, requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "mask": mask,
            "result": result
        }
        return result, context
    
    def vjp0(self):
        input_0, mask, result = self.context.values()
        grad_masked = result.grad.copy()
        grad_masked[mask.data] = 0
        input_0.grad += grad_masked


class Log(UnaryFunction):

    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        input_0, = input_tensors
        result = Tensor(data=np.log(input_0.data), requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "result": result
        }
        return result, context
    
    def vjp0(self):
        input_0, result = self.context.values()
        input_0.grad += result.grad / input_0.data


class Exp(UnaryFunction):
    
    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        input_0, = input_tensors
        result = Tensor(data=np.exp(input_0.data), requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "result": result
        }
        return result, context
    
    def vjp0(self):
        input_0, result = self.context.values()
        input_0.grad += result.grad * result.data


class Sum(UnaryFunction):

    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        
        input_0, = input_tensors
        
        if "dim" not in kwargs:
            raise ValueError("Sum requires 'dim' argument")
        dim = kwargs["dim"]
        if len(input_0.shape) > 2:
            raise NotImplementedError("I haven't implemented Sum for tensors of dim > 2 yet")
        
        result_data = np.sum(input_0.data, axis=dim, keepdims=True)
        result = Tensor(data=result_data, requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "result": result
        }
        return result, context
    
    def vjp0(self):
        input_0, result = self.context.values()
        grad = np.ones_like(input_0.data) * result.grad
        input_0.grad += grad


class LogSumExp(UnaryFunction):

    @staticmethod
    def forward(*input_tensors: Tensor, result_requires_grad: bool, **kwargs) -> tuple[Tensor, dict[str, Tensor]]:
        
        input_0, = input_tensors
        
        if "dim" not in kwargs:
            raise ValueError("Sum requires 'dim' argument")
        dim = kwargs["dim"]
        if len(input_0.shape) > 2:
            raise NotImplementedError("I haven't implemented LogSumExp for tensors of dim > 2 yet")
        
        data_max = np.max(input_0.data, axis=dim, keepdims=True)
        data_shifted = input_0.data - data_max
        exp_data_shifted = np.exp(data_shifted)
        sum_exp = np.sum(exp_data_shifted, axis=dim, keepdims=True)
        result_data = np.log(sum_exp) + data_max
        result = Tensor(data=result_data, requires_grad=result_requires_grad)
        context = {
            "input_0": input_0,
            "exp_data_shifted": Tensor(exp_data_shifted),
            "sum_exp": Tensor(sum_exp),
            "result": result
        }
        return result, context
    
    def vjp0(self):
        softmax = self.context["exp_data_shifted"].data / self.context["sum_exp"].data
        self.context["input_0"].grad += self.context["result"].grad * softmax

