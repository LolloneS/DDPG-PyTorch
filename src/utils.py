import torch

from src.variables import variables

device = variables["device"]


def to_tensor_variable(array, requires_grad=False):
    """Convert array to Tensor to the "best" available device."""
    return torch.tensor(array, requires_grad=requires_grad).to(device)
