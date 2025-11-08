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

@pytest.fixture(scope="function")
def alpha():
    return Tensor(3, requires_grad=True)


def test_basic_matrix_vector_multiply(W1, x1):
    y = W1 @ x1 
    assert np.allclose(y.data, np.array([5, 11], dtype=np.float32))
    y.backward()
    assert np.allclose(W1.grad, np.array([[1, 2], [1, 2]], dtype=np.float32))
    assert np.allclose(x1.grad, np.array([4, 6], dtype=np.float32))

def test_basic_vector_vector_add(b1, x1):
    y = b1 + x1
    assert np.allclose(y.data, np.array([5, 3], dtype=np.float32))
    y.backward()
    assert np.allclose(b1.grad, np.array([1, 1], dtype=np.float32))
    assert np.allclose(x1.grad, np.array([1, 1], dtype=np.float32))

def test_basic_vector_vector_sub(b1, x1):
    y = b1 - x1 
    assert np.allclose(y.data, np.array([3, -1], dtype=np.float32))
    y.backward()
    assert np.allclose(b1.grad, np.array([1, 1], dtype=np.float32))
    assert np.allclose(x1.grad, np.array([-1, -1], dtype=np.float32))

def test_basic_scalar_matrix_multiplication(W1, alpha):
    y1 = alpha * W1
    assert np.allclose(y1.data, np.array([[3, 6], [9, 12]], dtype=np.float32))
    y1.backward()
    assert np.allclose(alpha.grad, 10)
    assert np.allclose(W1.grad, np.array([[alpha.data, alpha.data], [alpha.data, alpha.data]], dtype=np.float32))

def test_basic_matrix_scalar_multiplication(W1, alpha):
    y1 = W1 * alpha
    assert np.allclose(y1.data, np.array([[3, 6], [9, 12]], dtype=np.float32))
    y1.backward()
    assert np.allclose(alpha.grad, 10)
    assert np.allclose(W1.grad, np.array([[alpha.data, alpha.data], [alpha.data, alpha.data]], dtype=np.float32))

def test_basic_relu(b2):
    y = b2.relu()
    assert np.allclose(y.data, np.array([0, 20, 0, 0, 1, 0], dtype=np.float32))
    y.backward()
    assert np.allclose(b2.grad, np.array([0, 1, 0, 0, 1, 0], dtype=np.float32))

    