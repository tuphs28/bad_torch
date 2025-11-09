from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

# Super basic implementation at the moment
# 
# TO-DO:
#   1) Handle data types
#   2) Handle different shapes + broadcasting // DONE
#   3) Make API less dogshit


class Tensor:

    def __init__(self, data: Union[np.ndarray, int, float], requires_grad: bool = False) -> None:

        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            assert data.dtype == np.float32, f"Expected float32 for numpy datatype, got {data.dtype}"
        else:
            raise ValueError(f"Tensor class does not currently support data of type {type(data)}")

        self.data = data
        self.grad = np.zeros_like(data)
        self.requires_grad = requires_grad

        self._backward_fncs = []
        self._parents = []


    @property
    def shape(self) -> tuple[int]:
        return self.data.shape
    
    def backward(self) -> None:

        self.grad = np.ones_like(self.data)

        visited_ancestors = set()
        ordered_ancestors = []
        def topo_sort(current_tensor):
            visited_ancestors.add(current_tensor)
            for parent in current_tensor._parents:
                if parent not in visited_ancestors:
                    topo_sort(parent)
            ordered_ancestors.append(current_tensor)
        topo_sort(self)

        for tensor in reversed(ordered_ancestors):
            for backward_fnc in tensor._backward_fncs:
                backward_fnc()

    def _unbroadcast_gradient(self, grad: np.ndarray) -> np.ndarray:

        ndims_added = len(grad.shape) - len(self.data.shape)
        if ndims_added > 0:
            grad = np.sum(grad, axis=tuple(range(ndims_added)))

        for dim_idx, (dim_grad, dim_data) in enumerate(zip(grad.shape, self.data.shape)):
            if dim_grad > 1 and dim_data == 1:
                grad = np.sum(grad, axis=dim_idx, keepdims=True)
        
        return grad


    def __add__(self, other: Union[int, float, Tensor]) -> Tensor:

        if isinstance(other, (int, float)):
            other = Tensor(other, requires_grad=False)
        elif not isinstance(other, Tensor):
            raise ValueError(f"Expected an int, a float, or Tensor, got a {type(other)}")
        
        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = self.data + other.data, 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]   

        if self.requires_grad:
            def vjp():
                grad = result.grad
                grad = self._unbroadcast_gradient(grad)
                self.grad += grad
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                grad = result.grad
                grad = other._unbroadcast_gradient(grad)
                other.grad += grad
            result._backward_fncs.append(vjp)

        return result
    

    def __sub__(self, other: Union[int, float, Tensor]) -> Tensor:

        if isinstance(other, (int, float)):
            other = Tensor(other, requires_grad=False)
        elif not isinstance(other, Tensor):
            raise ValueError(f"Expected an int, a float, or Tensor, got a {type(other)}")

        return self + (Tensor(-1) * other)
    

    def __mul__(self, other: Union[int, float, Tensor]) -> Tensor:

        if isinstance(other, (int, float)):
            other = Tensor(other, requires_grad=False)
        elif not isinstance(other, Tensor):
            raise ValueError(f"Expected an int, a float, or Tensor, got a {type(other)}")

        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = self.data * other.data, 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]   
    
        if self.requires_grad:
            def vjp():
                grad = result.grad * other.data
                grad = self._unbroadcast_gradient(grad)
                self.grad += grad
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                grad = result.grad * self.data
                grad = other._unbroadcast_gradient(grad)
                other.grad += grad
            result._backward_fncs.append(vjp)

        return result
    
    def __matmul__(self, other: Tensor) -> Tensor:

        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)} which is not currently supported"

        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = self.data @ other.data, 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]   

        if self.requires_grad:
            def vjp():
                other_data = other.data if len(other.data.shape) > 1 else other.data[:,None]
                result_grad = result.grad if len(result.grad.shape) > 1 else result.grad[:,None]
                grad = result_grad @ other_data.swapaxes(-1, -2)
                grad = grad if len(self.data.shape) > 1 else grad[:,0]
                self.grad += grad
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                self_data = self.data if len(self.data.shape) > 1 else self.data[:,None]
                result_grad = result.grad if len(result.grad.shape) > 1 else result.grad[:,None]
                grad = self_data.swapaxes(-1, -2) @ result_grad
                grad = grad if len(other.data.shape) > 1 else grad[:,0]
                other.grad += grad
            result._backward_fncs.append(vjp)

        return result

    
    def relu(self) -> Tensor:
        """Temporary method whilst working on code. Here, self in a nx1 vector"""

        mask = self.data <= 0.0
        data_masked = self.data.copy()
        data_masked[mask] = 0
        result = Tensor(
            data = data_masked, 
            requires_grad = self.requires_grad
        )
        result._parents += [self]

        if self.requires_grad:
            def make_vjp(mask):
                def vjp():
                    grad_masked = result.grad.copy()
                    grad_masked[mask] = 0
                    self.grad += grad_masked
                return vjp
            vjp = make_vjp(mask)
            result._backward_fncs.append(vjp)

        return result
    
    def log(self) -> Tensor:
        result = Tensor(
            data = np.log(self.data), 
            requires_grad = self.requires_grad
        )
        result._parents += [self]

        if self.requires_grad:
            def vjp():
                self.grad += result.grad / self.data
            result._backward_fncs.append(vjp)

        return result
    
    def exp(self) -> Tensor:
        result = Tensor(
            data = np.exp(self.data), 
            requires_grad = self.requires_grad
        )
        result._parents += [self]

        if self.requires_grad:
            def vjp():
                self.grad += result.grad * self.data
            result._backward_fncs.append(vjp)

        return result

