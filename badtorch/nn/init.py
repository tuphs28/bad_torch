import numpy as np

from badtorch.autograd import Tensor

# I'm following torch convention here for init to be inplace - not sure why it does this?

def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> None:

    if len(tensor.shape) < 2:
        raise ValueError("Xavier init require a tensor with >= 2 dims")
    
    fan_in, fan_out = tensor.shape[0], tensor.shape[1] # only doing linear atm, change if needed
    bound = gain * np.sqrt(6 / (fan_in + fan_out))
    tensor.data = np.random.uniform(low=-bound, high=bound, size=tensor.shape)

def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> None:

    if len(tensor.shape) < 2:
        raise ValueError("Xavier init require a tensor with >= 2 dims")
    
    fan_in, fan_out = tensor.shape[0], tensor.shape[1] # only doing linear atm, change if needed
    sigma = gain * np.sqrt(2 / (fan_in + fan_out))
    tensor.data = np.random.normal(loc=0, scale=sigma, size=tensor.shape)
