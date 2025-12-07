from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from functions import Function


class Tensor:

    def __init__(self, data: Union[np.ndarray, int, float], requires_grad: bool = False) -> None:

        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Tensor class does not currently support data of type {type(data)}")

        self.data = data
        self.grad = np.zeros_like(data)
        self.requires_grad = requires_grad

        self._backward_fncs = []

        self._creator: Optional[Function] = None

    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    def backward(self) -> None:
        self.grad = np.ones_like(self.data)
        visited_ancestors = set()
        ordered_ancestors = []
        def topo_sort(current_tensor: Tensor):
            visited_ancestors.add(current_tensor)
            if current_tensor._creator:
                for parent in current_tensor._creator.parents:
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
        from .functions import Add
        result = Add.apply(self, other)
        return result 

    def __mul__(self, other: Union[int, float, Tensor]) -> Tensor:
        from .functions import Mul
        result = Mul.apply(self, other)
        return result
    
    def __neg__(self) -> Tensor:
        return self * -1
    
    def __sub__(self, other: Union[int, float, Tensor]) -> Tensor:
        return self + (-other)
    
    def __matmul__(self, other: Union[int, float, Tensor]) -> Tensor:
        from .functions import MatMul
        result = MatMul.apply(self, other)
        return result

    def relu(self) -> Tensor:
        from .functions import ReLU
        result = ReLU.apply(self)
        return result
    
    def log(self) -> Tensor:
        from .functions import Log
        result = Log.apply(self)
        return result
    
    def exp(self) -> Tensor:
        from .functions import Exp
        result = Exp.apply(self)
        return result
    
    def sum(self, dim: int) -> Tensor:
        from .functions import Sum
        result = Sum.apply(self, dim=dim)
        return result
    
    def logsumexp(self, dim: int) -> Tensor:
        from .functions import LogSumExp
        result = LogSumExp.apply(self, dim=dim)
        return result

