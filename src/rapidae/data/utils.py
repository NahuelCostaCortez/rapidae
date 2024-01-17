"""
Class for common data utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from inspect import getmembers, isfunction
from rapidae.conf import Logger


def plot_label_clusters(model, data, labels):
    """
    Takes a trained variational autoencoder (VAE) model, input data, and true labels.
    It uses the VAE model to predict the latent space representations for the input data and then
    creates a 2D scatter plot of the latent space clusters colored by the true labels.

    Args:
        model (keras.Model): An instance of a trained Variational Autoencoder (VAE) model.
            The model should have a 'predict' method that can generate predictions for the given input data.
        data (numpy.ndarray): Input data to be used for generating latent space representations.
            It should be compatible with the input shape expected by the VAE model.
        labels (numpy.ndarray): True labels corresponding to the input data.
            It should have the same length as the number of samples in the input data.  
    """
    outputs = model.predict(data, verbose=0)

    plt.figure(figsize=(12, 10))
    plt.scatter(outputs['z_mean'][:, 0], outputs['z_mean'][:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def evaluate(y_true, y_hat, sel_metric, label='test'):
    """
    Takes ground truth and predicted values along with a selected evaluation metric and calculates the
    performance of the model based on the given metric. It supports both custom metrics from AEPY and metrics from the
    scikit-learn library.
    The selected metric must be compatible with the format of y_true and y_hat.

    Args:
        y_true (ArrayLike): Ground truth (correct) target values.
        y_hat (ArrayLike): Estimated target values.
        sel_metric (callable): The selected metric, either a custom metric from AEPY or a metric from scikit-learn.
        label (str): Tag of the evaluated set (e.g., 'test', 'validation').

    Returns:
        result (ArrayLike): The result of the evaluation based on the selected metric.
    """
    # Get all functions from the metrics module
    all_metrics = getmembers(metrics, isfunction)

    # Filter out private functions
    all_metrics = [metric[0]
                   for metric in all_metrics if not metric[0].startswith('_')]

    # Check if the selected metric is available in sklearn
    if type(sel_metric).__name__ == 'function' and sel_metric.__name__ in all_metrics:
        Logger().log_info('Using Scikit-learn metric...')
        result = sel_metric(y_true, y_hat)
        print('{} set results: [\n\t {}: {} \n]'.format(
            label, sel_metric.__name__, result))
    else:
        Logger().log_info('Using Rapidae custom metric...')
        result = sel_metric.calculate(y_true, y_hat)
        print('{} set results: [\n\t {}: {} \n]'.format(
            label, sel_metric.__class__.__name__, result))

    return result


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


def display_diff(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    Useful for checking differences in processed images.
    Source: https://keras.io/examples/vision/autoencoder/

    Args:
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
