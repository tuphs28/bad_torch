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
    def shape(self):
        return self.data.shape
    
    def backward(self):

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


    def __add__(self, other):

        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)} which is not currently supported"

        if (len(self.shape) == 1) and (len(other.shape) == 1) and (self.shape[0] == other.shape[0]):
            result = self._vector_vector_add(other)
        else:
            raise NotImplementedError("Tensor addition for a {self.shape} Tensor with a {other.shape} Tensor not currently implemented")

        return result
    

    def __sub__(self, other):

        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)} which is not currently supported"

        return self + (Tensor(-1, requires_grad=other.requires_grad) * other)
    

    def __mul__(self, other):

        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)} which is not currently supported"

        if (len(self.shape) == 0) and (len(other.shape) == 1): # scalar-vector multiply
            result = self._scalar_vector_multiply(other)
        elif (len(self.shape) == 1) and (len(other.shape) == 0): # vector-scalar multiply
            result = other._scalar_vector_multiply(self)
        elif (len(self.shape) == 0) and (len(other.shape) == 2): # scalar-matrix multiply
            result = self._scalar_matrix_multiply(other)
        elif (len(self.shape) == 2) and (len(other.shape) == 0): # scalar-matrix multiply
            result = other._scalar_matrix_multiply(self)
        else:
            raise NotImplementedError(f"Multiplication for a {self.shape} Tensor with a {other.shape} Tensor not currently implemented")
        
        return result
    
    def __matmul__(self, other):

        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)} which is not currently supported"

        if (len(self.shape) == 2) and (len(other.shape) == 1) and (self.shape[-1] == other.shape[0]): # Matrix-vector multiply
            result = self._matrix_vector_matrixmultiply(other)
        elif (len(self.shape) == 1) and (len(other.shape) == 1) and (self.shape[0] == other.shape[0]): # Vector-vector multiply 
            result = self._vector_vector_matrixmultiply(other)
        else:
            raise NotImplementedError("Tensor multiply for a {self.shape} Tensor with a {other.shape} Tensor not currently implemented")
        
        return result
    

    def _matrix_vector_matrixmultiply(self, other):
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
    

    def _vector_vector_matrixmultiply(self, other):
        """Temporary method whilst working on code. Here, self in a mxn matrix and other is a nx1 vector"""
        
        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)}"

        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = np.dot(self.data, other.data), 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]

        if self.requires_grad:
            def vjp():
                self.grad += result.grad * other.data
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                other.grad += result.grad * self.data
            result._backward_fncs.append(vjp)

        return result
    

    def _scalar_vector_multiply(self, other):
        """Temporary method whilst working on code. Here, self is a scalar (e.g. a 0-dim Tensor) and other is a nx1 vector"""
        
        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)}"
        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = self.data * other.data, 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]   
    
        if self.requires_grad:
            def vjp():
                self.grad +=  np.dot(result.grad, other.data)
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                other.grad += result.grad * self.data 
            result._backward_fncs.append(vjp)

        return result
    
    def _scalar_matrix_multiply(self, other):
        """Temporary method whilst working on code. Here, self is a scalar (e.g. a 0-dim Tensor) and other is a nx1 vector"""
        
        assert isinstance(other, Tensor), f"Expected a Tensor, got a {type(other)}"
        result_requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            data = self.data * other.data, 
            requires_grad = result_requires_grad
        )
        result._parents += [self, other]   
    
        if self.requires_grad:
            def vjp():
                self.grad += np.sum(result.grad * other.data)
            result._backward_fncs.append(vjp)
        if other.requires_grad:
            def vjp():
                other.grad += result.grad * self.data
            result._backward_fncs.append(vjp)

        return result

    def _vector_vector_add(self, other):
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

