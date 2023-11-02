"""
Class for common data utilities.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from inspect import getmembers, isfunction
from conf import RandomSeed, Logger


def viz_latent_space(exp_name, model, data, targets=[], epoch='Final', save=False, show=False, path='./'):
    """
    Displays a plot of the latent space

    Args
    ----
        exp_name (str): Name of the experiment.
        model (keras.Model): Used model.
        data ():
        targets ():
        epoch (str):
        save (bool): Flag for saving the generated plot.
        show (bool): Flag for showing the generated plot.
        path (str): Path where the plot will be stored.
    """
    z = model.call(dict(data=data))['z']

    plt.figure(figsize=(8, 10))
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
    Evaluates model results with a selected metric.

    Args
    ----
        y_true (ArrayLike): Ground truth (correct) target values.
        y_hat (ArrayLike): Estimated target values.
        sel_metric (callable): Class name of the selected metric (AEPY's own) or sklearn metric
        label (str): Tag of the evaluated set.
    """
    logger = Logger()

    # Get all functions from the metrics module
    all_metrics = getmembers(metrics, isfunction)

    # Filter out private functions
    all_metrics = [metric[0] for metric in all_metrics if not metric[0].startswith('_')]

    # Check if the selected metric is available in sklearn
    if type(sel_metric).__name__=='function' and sel_metric.__name__ in all_metrics:
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
    Sets random seed for allow experiment reproducibility.

    Args
    ----
        random_seed (int): Selected random seed.
    """
    # Set Random seed of an experiment
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def plot_mnist_images(images, labels, num_images=10):
    """
    Visualizes MNIST images and their corresponding labels.

    Args:
        images (numpy.ndarray): An array of MNIST images.
        labels (numpy.ndarray): An array of corresponding labels.
        num_images (int): Number of images to display. Default is 10.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))

    for i in range(num_images):
        # Display the image
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')

    plt.show()


def add_noise(array, noise_factor=0.4, a_min=0, a_max=1):
    """
    Adds random noise to each image in the supplied array.
    """
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    Source: https://keras.io/examples/vision/autoencoder/
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
