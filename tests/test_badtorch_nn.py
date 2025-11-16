import pytest

from badtorch.nn import MLP


def test_simple_mlp_param_count():

    model = MLP(
        in_dims=[10, 20],
        out_dims=[20, 30],
        dropout=False
    )

    assert model.param_count() == (10*20 + 20 + 20 * 30 + 30)