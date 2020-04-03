import numpy as np


def noise(shape=None, mu=0.0, sigma=0.1):
    """Create Gaussian noise from a normal distribution
    with mean `mu` and stdev `sigma`."""
    noise = np.random.normal(loc=mu, scale=sigma, size=shape)
    return noise
