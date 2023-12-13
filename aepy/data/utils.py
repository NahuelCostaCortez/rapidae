"""
Class for common data utilities.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from inspect import getmembers, isfunction
from aepy.conf import RandomSeed, Logger


def viz_latent_space(exp_name, model, data, targets=[], epoch='Final', save=False, show=False, path='./'):
    """
    Displays a plot of the latent space. The plot can be customized by providing targets for coloring data points,
    specifying the training epoch, and choosing whether to save or display the plot.

    Args
    ----
        exp_name (str): Name of the experiment.
        model (keras.Model): Used model.
        data (ArrayLike): The input data for which the latent space is visualized.
        targets (ArrayLike): Optional targets for coloring data points in the plot.
        epoch (str): The epoch or stage of the model to be visualized (default is 'Final').
        save (bool): Flag for saving the generated plot.
        show (bool): Flag for showing the generated plot.
        path (str): Path where the plot will be stored.

    Returns
    -------
    z (numpy.ndarray): The latent space representation of the input data.
    """
    # Obtain the latent space representation from the model
    z = model.call(dict(data=data))['z']

    plt.figure(figsize=(8, 10))

    # Check if targets are provided for coloring data points
    if len(targets) > 0:
        plt.scatter(z[:, 0], z[:, 1], c=targets)
    else:
        plt.scatter(z[:, 0], z[:, 1])

    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()

    if show:
        plt.show()
    if save:
        plt.savefig(path+exp_name+'_latent_space.png')

    return z


def evaluate(y_true, y_hat, sel_metric, label='test'):
    """
    Takes ground truth and predicted values along with a selected evaluation metric and calculates the
    performance of the model based on the given metric. It supports both custom metrics from AEPY and metrics from the
    scikit-learn library.
    The selected metric must be compatible with the format of y_true and y_hat.

    Args
    ----
        y_true (ArrayLike): Ground truth (correct) target values.
        y_hat (ArrayLike): Estimated target values.
        sel_metric (callable): The selected metric, either a custom metric from AEPY or a metric from scikit-learn.
        label (str): Tag of the evaluated set (e.g., 'test', 'validation').

    Returns
    -------
    result: The result of the evaluation based on the selected metric.
    """
    logger = Logger()

    # Get all functions from the metrics module
    all_metrics = getmembers(metrics, isfunction)

    # Filter out private functions
    all_metrics = [metric[0]
                   for metric in all_metrics if not metric[0].startswith('_')]

    # Check if the selected metric is available in sklearn
    if type(sel_metric).__name__ == 'function' and sel_metric.__name__ in all_metrics:
        result = sel_metric(y_true, y_hat)
        print('{} set results: [\n\t {}: {} \n]'.format(
            label, sel_metric.__name__, result))
    else:
        result = sel_metric.calculate(y_true, y_hat)
        print('{} set results: [\n\t {}: {} \n]'.format(
            label, sel_metric.__class__.__name__, result))

    return result


def set_random_seed(random_seed=RandomSeed.RANDOM_SEED):
    """
    This function ensures that the randomness introduced by random number generation functions is controlled, providing
    a reproducible environment for experiments.

    Args
    ----
        random_seed (int): Selected random seed.
    """
    # Set Random seed of an experiment
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def add_noise(array, noise_factor=0.4, a_min=0, a_max=1):
    """
    Adds random noise to each the supplied array to simulate noise or variability in data.
    The noise is generated from a normal distribution with a specified noise factor. 
    The resulting noisy array is then clipped to ensure that values fall within the specified range [a_min, a_max].

    Args
    ----
        array (numpy.ndarray): The input array to which noise will be added.
        noise_factor (float): The magnitude of the random noise to be added (default is 0.4).
        a_min (float): The minimum value after adding noise (default is 0).
        a_max (float): The maximum value after adding noise (default is 1).

    Returns
    -------
        numpy.ndarray: The array with added noise, clipped to the specified range.
    """
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display_diff(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    Useful for checking differences in processed images.
    Source: https://keras.io/examples/vision/autoencoder/

    Args
    ----
        array1 (numpy.ndarray): The first array containing images for comparison.
        array2 (numpy.ndarray): The second array containing images for comparison.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
