"""
Class for common data utilities.
"""

import os
import gzip
import urllib
import numpy as np
from shutil import rmtree

from rapidae.conf import Logger


def get_data_from_url(url, filename, data_dir, persistant=False):
    """
    Download data from a specific url.

    Args:
        url (str): Given url from where the data will be downloaded.
        filename (str): Name of the file to be downloaded.
        data_dir (str): Directory where the data will be stored.
        persistant (bool): If True, keeps the downloaded dataset files.
    Returns:
        data (dict): Dictionary containing the data.
    """
    # Create a directory to store the downloaded files
    os.makedirs(data_dir, exist_ok=True)

    target_path = os.path.join(data_dir, filename)

    if not os.path.exists(target_path):
        Logger().log_info(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, target_path)
    else:
        Logger().log_info(f"Skipping... Data already exists.")

    data = np.load(target_path, allow_pickle=True).item()

    # If required delete data
    if not persistant:
        Logger().log_info("Deleting data...")
        rmtree(data_dir)

    return data


def mnist_load_images(filename):
    """
    Auxiliary function to load gzipped images for MNIST dataset.

    Args:
        filename (str): Path to the file.
    """
    with gzip.open(filename, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)

    return data.reshape(-1, 28, 28, 1)


def mnist_load_labels(filename):
    """
    Auxiliary function to load gzipped labels for MNIST dataset.

    Args:
        filename (str): Path to the file.
    """
    with gzip.open(filename, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)

    return data


def convert_one_hot(x, target, n_classes=6):
    """
    Convert target values to one-hot encoding.

    Args:
        x (numpy.ndarray): Input array or matrix.
        target (numpy.ndarray): Target values to be converted to one-hot encoding.
        n_classes (int): Number of classes in the target values.

    Returns:
        samples (numpy.ndarray): Array with one-hot encoded representation of target values.
    """
    samples = np.zeros((x.shape[0], n_classes))

    for i in range(target.shape[0]):
        samples[i][int(target[i])] = 1

    return samples


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
