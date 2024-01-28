"""
Class for common data utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def add_noise(array, noise_factor=0.4, a_min=0, a_max=1):
    """
    Adds random noise to each the supplied array to simulate noise or variability in data.
    The noise is generated from a normal distribution with a specified noise factor.
    The resulting noisy array is then clipped to ensure that values fall within the specified range [a_min, a_max].

    Args:
        array (numpy.ndarray): The input array to which noise will be added.
        noise_factor (float): The magnitude of the random noise to be added (default is 0.4).
        a_min (float): The minimum value after adding noise (default is 0).
        a_max (float): The maximum value after adding noise (default is 1).

    Returns:
        array (numpy.ndarray): The array with added noise, clipped to the specified range.
    """
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)
