import numpy as np
from sklearn.datasets import fetch_openml

from badtorch.autograd import Tensor

def load_mnist_data(num_total: int, pct_test: float = 0.1):

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X, y_raw = mnist["data"][:num_total] / 255.0, mnist["target"][:num_total]
    y = np.zeros((y_raw.shape[0], 10))
    y[np.arange(y_raw.shape[0]), y_raw.astype(int)] = 1
    num_train = num_total - int(num_total * pct_test)
    X_tr, X_te = X[:num_train], X[num_train:]
    y_tr, y_te = y[:num_train], y[num_train:]

    return X_tr, y_tr, X_te, y_te

class DataLoader:

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            perm = np.random.permutation(self.num_samples)
            self.X = self.X[perm]
            self.y = self.y[perm]
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_X = self.X[start_idx:end_idx]
            batch_y = self.y[start_idx:end_idx]
            self.current_batch += 1
            return Tensor(batch_X), Tensor(batch_y)
        else:
            raise StopIteration
