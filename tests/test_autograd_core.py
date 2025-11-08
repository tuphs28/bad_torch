import numpy as np
import pytest

from badtorch.autograd.core import Tensor


@pytest.fixture(scope="function")
def W1():
    return Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True)

@pytest.fixture(scope="function")
def x1():
    return Tensor(np.array([1,2], dtype=np.float32), requires_grad=True)

@pytest.fixture(scope="function")
def b1():
    return Tensor(np.array([4,1], dtype=np.float32), requires_grad=True) 

@pytest.fixture(scope="function")
def b2():
    return Tensor(np.array([-10, 20, -3, 0, 1, -1], dtype=np.float32), requires_grad=True) 


def test_basic_matrix_vector_multiply(W1, x1):
    y = W1.matrix_vector_multiply(x1) 
    assert np.allclose(y.data, np.array([5, 11], dtype=np.float32))
    y.backward()
    assert np.allclose(W1.grad, np.array([[1, 2], [1, 2]], dtype=np.float32))
    assert np.allclose(x1.grad, np.array([4, 6], dtype=np.float32))

def test_basic_vector_vector_add_multiply(b1, x1):
    y = b1.vector_vector_add(x1) 
    assert np.allclose(y.data, np.array([5, 3], dtype=np.float32))
    y.backward()
    assert np.allclose(b1.grad, np.array([1, 1], dtype=np.float32))
    assert np.allclose(x1.grad, np.array([1, 1], dtype=np.float32))

def test_basic_relu(b2):
    y = b2.relu()
    assert np.allclose(y.data, np.array([0, 20, 0, 0, 1, 0], dtype=np.float32))
    y.backward()
    assert np.allclose(b2.grad, np.array([0, 1, 0, 0, 1, 0], dtype=np.float32))

    