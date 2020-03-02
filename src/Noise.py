import torch
from torch.distributions.normal import Normal

from src.variables import variables

device = variables["device"]


mu = torch.tensor([0.0])
sigma = torch.tensor([0.1])


def noise(shape=None):
    """Create Gaussian noise from a normal distribution with mean 0 and stdev 0.5."""
    noise = Normal(loc=mu, scale=sigma).sample(shape).view(1, -1).to(device)
    return noise
