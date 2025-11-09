
import numpy as np
from typing import Literal

from badtorch.autograd import Tensor
from badtorch.nn import MLP
from badtorch.optim import SGD, Adam
from badtorch.functional import cross_entropy_loss
from badtorch.training.data import DataLoader, load_mnist_data

def train_mlp(
    num_epochs: int = 10,
    in_dims: list[int] = [784, 256, 128],
    out_dims: list[int] = [256, 128, 10],
    dropout: bool = True,
    dropout_p: float = 0.05,
    optimiser_name: Literal["sgd", "adam"] = "sgd",
    lr: float = 1e-3,
    batch_size: int = 32,
    num_total: int = 10000,
    pct_test: float = 0.1
) -> dict:
    

    X_tr, y_tr, X_te, y_te = load_mnist_data(num_total=num_total, pct_test=pct_test)
    train_loader = DataLoader(X_tr, y_tr, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_te, y_te, batch_size=batch_size, shuffle=False)
    
    model = MLP(
        in_dims=in_dims,
        out_dims=out_dims,
        dropout=dropout,
        dropout_p=dropout_p
    )

    if optimiser_name == "sgd":
        optimiser = SGD(parameters=model.parameters(), lr=lr)
    elif optimiser_name == "adam":
        optimiser = Adam(parameters=model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unrecognised optimiser name: {optimiser_name}")



    results = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(num_epochs):

        model.train_mode()
        train_loss = 0.0
        num_batches = 0
        for Xb, Yb in train_loader:
            logits = model(Xb)
            loss = cross_entropy_loss(logits, Yb, dim=1)
            loss = loss * Tensor(1.0 / Xb.shape[0])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.data.sum().item()
            num_batches += 1
        avg_train_loss = train_loss / num_batches

        model.eval_mode()
        test_loss = 0.0
        test_acc = 0.0
        num_batches = 0
        for Xb, Yb in test_loader:
            logits = model(Xb)
            loss = cross_entropy_loss(logits, Yb, dim=1)
            loss = loss * Tensor(1.0 / Xb.shape[0])
            test_loss += loss.data.sum().item()
            preds = np.argmax(logits.data, axis=1)
            true_labels = np.argmax(Yb.data, axis=1)
            test_acc += np.sum(preds == true_labels)
        avg_test_loss = test_loss / test_loader.num_batches
        avg_test_acc = test_acc / len(X_te)

        results["train_loss"].append(avg_train_loss)
        results["test_loss"].append(avg_test_loss)
        results["test_acc"].append(avg_test_acc)

    return results