import numpy as np
import pytest

from badtorch.autograd.core import Tensor

@pytest.fixture(scope="function")
def W1_4x4():
    return Tensor(np.array([[1, 1, 1, 1],
                            [2, 2, 2, 2],
                            [3, 3, 3, 3],
                            [4, 4, 4, 4]], dtype=np.float32), requires_grad=True)

@pytest.fixture(scope="function")
def W2_4x4():
    return Tensor(np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32), requires_grad=True)

@pytest.fixture(scope="function")
def b_4():
    return Tensor(np.array([1, 2, 3, 4], dtype=np.float32), requires_grad=True)

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

def test_matrix_multiply_with_bias_broadcasting(W1_4x4, W2_4x4, b_4):
    Y = W1_4x4 @ W2_4x4 + b_4
    expected_Y = np.array([[2, 3, 4, 5],
                           [3, 4, 5, 6],
                           [4, 5, 6, 7],
                           [5, 6, 7, 8]], dtype=np.float32)
    assert np.allclose(Y.data, expected_Y)
    Y.backward()
    
    expected_W1_grad = np.ones((4, 4), dtype=np.float32)
    assert np.allclose(W1_4x4.grad, expected_W1_grad)
    expected_W2_grad = np.full((4, 4), 10.0, dtype=np.float32)
    assert np.allclose(W2_4x4.grad, expected_W2_grad)
    expected_b_grad = np.array([4, 4, 4, 4], dtype=np.float32)
    assert np.allclose(b_4.grad, expected_b_grad)

def test_matrix_multiply_with_scalar_bias(W1_4x4, W2_4x4):
    b_scalar = Tensor(10.0, requires_grad=True)
    Y = W1_4x4 @ W2_4x4 + b_scalar
    expected_Y = np.array([[11, 11, 11, 11],
                           [12, 12, 12, 12],
                           [13, 13, 13, 13],
                           [14, 14, 14, 14]], dtype=np.float32)
    assert np.allclose(Y.data, expected_Y)
    Y.backward()
    assert np.allclose(b_scalar.grad, 16.0)
    assert np.allclose(W1_4x4.grad, np.ones((4, 4), dtype=np.float32))
    assert np.allclose(W2_4x4.grad, np.full((4, 4), 10.0, dtype=np.float32))

def test_matrix_vector_multiply_with_bias():
    W = Tensor(np.array([[1, 2, 3],
                         [4, 5, 6]], dtype=np.float32), requires_grad=True)
    x = Tensor(np.array([1, 1, 1], dtype=np.float32), requires_grad=True)
    b = Tensor(np.array([10, 20], dtype=np.float32), requires_grad=True)
    y = W @ x + b
    expected_y = np.array([16, 35], dtype=np.float32)  
    assert np.allclose(y.data, expected_y)
    y.backward()
    assert np.allclose(x.grad, np.array([5, 7, 9], dtype=np.float32))
    assert np.allclose(W.grad, np.ones((2, 3), dtype=np.float32))
    assert np.allclose(b.grad, np.array([1, 1], dtype=np.float32))

def test_matrix_vector_multiply_with_bias_and_relu():
    W = Tensor(np.array([[1, 2, 3],
                         [-4, -5, -6]], dtype=np.float32), requires_grad=True)
    x = Tensor(np.array([1, 1, 1], dtype=np.float32), requires_grad=True)
    b = Tensor(np.array([10, -5], dtype=np.float32), requires_grad=True)
    y = W @ x + b
    z = y.relu()
    expected_z = np.array([16, 0], dtype=np.float32)
    assert np.allclose(z.data, expected_z)
    z.backward()
    assert np.allclose(x.grad, np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(W.grad, np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32))
    assert np.allclose(b.grad, np.array([1, 0], dtype=np.float32))

def test_log_forward():
    x = Tensor(np.array([1.0, np.e, np.e**2], dtype=np.float32), requires_grad=True)
    y = x.log()
    assert np.allclose(y.data, [0.0, 1.0, 2.0], atol=1e-6)

def test_log_backward():
    x = Tensor(np.array([2.0, 4.0], dtype=np.float32), requires_grad=True)
    y = x.log()
    y.backward()
    assert np.allclose(x.grad, [0.5, 0.25])  

def test_exp_forward():
    x = Tensor(np.array([0.0, 1.0, 2.0], dtype=np.float32), requires_grad=True)
    y = x.exp()
    assert np.allclose(y.data, [1.0, np.e, np.e**2], atol=1e-6)

def test_exp_backward():
    x = Tensor(np.array([0.0, 1.0, 2.0], dtype=np.float32), requires_grad=True)
    y = x.exp()
    y.backward()
    assert np.allclose(x.grad, np.exp(x.data), atol=1e-6)

def test_sum_1d():
    x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    y = x.sum(dim=0)
    assert np.allclose(y.data, 6.0)
    y.backward()
    assert np.allclose(x.grad, [1.0, 1.0, 1.0])

def test_sum_2d_along_cols():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    y = x.sum(dim=1)
    assert y.shape == (2, 1)  
    assert np.allclose(y.data, [[3.0], [7.0]])
    y.backward()
    assert np.allclose(x.grad, [[1.0, 1.0], [1.0, 1.0]])

def test_sum_2d_along_rows():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    y = x.sum(dim=0)
    assert y.shape == (1, 2)  
    assert np.allclose(y.data, [[4.0, 6.0]])
    y.backward()
    assert np.allclose(x.grad, [[1.0, 1.0], [1.0, 1.0]])

def test_logsumexp_numerical_stability():
    x = Tensor(np.array([1000.0, 1001.0, 1002.0], dtype=np.float32), requires_grad=True)
    y = x.logsumexp(dim=0)
    assert np.isfinite(y.data)
    assert np.allclose(y.data, 1002.407, atol=0.01)

def test_logsumexp_backward_is_softmax():
    x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    y = x.logsumexp(dim=0)
    y.backward()
    exp_x = np.exp(x.data)
    expected_grad = exp_x / np.sum(exp_x)
    assert np.allclose(x.grad, expected_grad, atol=1e-6)

def test_logsumexp_2d():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    y = x.logsumexp(dim=1)
    assert y.shape == (2, 1)
    y.backward()
    assert np.allclose(x.grad.sum(axis=1, keepdims=True), [[1.0], [1.0]])
