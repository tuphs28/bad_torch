from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

# Super basic implementation at the moment
# 
# TO-DO:
#   1) Handle data types
#   2) Handle different shapes + broadcasting
#   3) Make API less dogshit


class Tensor:

    def __init__(self, data, requires_grad = False):

        # Temporarily: force all input data to be float32 numpy arrays
        assert isinstance(data, np.ndarray), f"Expected numpy array, got {type(data)}"
        assert data.dtype == np.float32, f"Expected float32 array, got {data.dtype}"

        self.data = data
        self.grad = np.zeros_like(data)
        self.requires_grad = requires_grad

        self._backward_fncs = []
        self._parents = []


    @property
    def shape(self):
        return self.data.shape
    
    def backward(self):

        self.grad = np.ones_like(self.data)

        visited_ancestors = []
        ordered_ancestors = []
        def topo_sort(current_tensor):
            visited_ancestors.append(current_tensor)
            for parent in current_tensor._parents:
                if parent not in visited_ancestors:
                    topo_sort(parent)
            ordered_ancestors.append(current_tensor)
        topo_sort(self)

        for tensor in reversed(ordered_ancestors):
            for backward_fnc in tensor._backward_fncs:
                backward_fnc()

    def matrix_vector_multiply(self, other):
        """Temporary method whilst working on code. Here, self in a mxn matrix and other is a nx1 vector"""
        
        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)}"

        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = self.data @ other.data, 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]

        if self.requires_grad:
            def vjp():
                self.grad += np.outer(result.grad, other.data)
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                other.grad += self.data.T @ result.grad
            result._backward_fncs.append(vjp)

        return result
    
    def vector_vector_add(self, other):
        """Temporary method whilst working on code. Here, self in a nx1 vector and other is a nx1 vector"""

        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)}"
        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = self.data + other.data, 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]   

        if self.requires_grad:
            def vjp():
                self.grad += result.grad
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                other.grad += result.grad
            result._backward_fncs.append(vjp)

        return result
    
    def relu(self):
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
            def vjp():
                grad_masked = result.grad.copy()
                grad_masked[mask] = 0
                self.grad += grad_masked
            result._backward_fncs.append(vjp)

        return result

