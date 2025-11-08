import numpy as np
import pytest

from badtorch.autograd.core import Tensor


@pytest.fixture(scope="function")
def W1():
    return Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), requires_grad=True)

@pytest.fixture(scope="function")
def x1():
    return Tensor(np.array([1,2], dtype=np.float32), requires_grad=True)

def test_basic_linear_matrix_vector_multiply(W1, x1):
    y = W1.matrix_vector_multiply(x1) 
    y.backward()
    assert np.allclose(W1.grad, np.array([[1, 2], [1, 2]], dtype=np.float32))
    assert np.allclose(x1.grad, np.array([4, 6], dtype=np.float32))
    