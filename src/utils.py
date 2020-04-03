import torch

from src.variables import variables

device = variables["device"]


def to_tensor_variable(array, requires_grad=False, dtype=torch.float32):
    """Convert array to Tensor to the "best" available device."""
    return torch.tensor(
        array, requires_grad=requires_grad, dtype=dtype, device=device
    )
