"""
Class for common data utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from inspect import getmembers, isfunction
from rapidae.conf import Logger


def plot_latent_space(
    z,
    labels=None,
    colormap="plasma",
    plot=True,
    save=False,
    path="./latent_space.png",
):
    """
    Plot latent space a 2D scatter plot.

    Args:
        z (numpy.ndarray): The latent space. If the shape is greater than 2, dimensionality reduction is performed using t-SNE.
        labels (numpy.ndarray, optional): The labels for each data point. If provided, the data points are colored based on their labels.
        colormap (str, optional): The colormap to use for the scatter plot.
        plot (bool, optional): Flag to display the plot. Defaults to True.
        save (bool, optional): Flag to save the plot. Defaults to False.
        path (str, optional): The path to save the plot. Defaults to './latent_space.png'.

    Returns:
        None
    """
    if z.shape[1] > 2:
        from sklearn.manifold import TSNE

        Logger().log_info(
            "Latent space > 2. Performing dimensionality reduction using t-SNE..."
        )
        model = TSNE(n_components=2, random_state=0)
        z = model.fit_transform(z)

    plt.figure(figsize=(12, 10))
    if labels is None:
        plt.scatter(z[:, 0], z[:, 1], cmap=colormap)
    else:
        plt.scatter(z[:, 0], z[:, 1], c=labels, cmap=colormap)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if save:
        plt.savefig(path)
    if plot:
        plt.show()
    # Close the plot to avoid memory leaks
    plt.close()


def plot_reconstructions(
    original,
    reconstructed,
    type="image",
    plot=True,
    save=False,
    path="./reconstructions.png",
    num_samples=10,
):
    """
    Displays ten random samples from each one of the supplied arrays.
    Useful for checking the quality of the reconstructions.

    Args:
        original (numpy.ndarray): The first array containing images for comparison.
        reconstructed (numpy.ndarray): The second array containing images for comparison.
        type (str): The type of data, either 'image' or 'ts' (time_series).
        plot (bool, optional): Flag to display the plot. Defaults to True.
        save (bool, optional): Flag to save the plot. Defaults to False.
        path (str, optional): The path to save the plot. Defaults to './reconstructions.png'.

    Returns:
        None
    """

    indices = np.random.randint(len(original), size=num_samples)
    samples1 = original[indices, :]
    samples2 = reconstructed[indices, :]

    if type == "image":
        if len(samples1.shape) == 2:
            width = int(np.sqrt(samples1.shape[1]))
            samples1 = samples1.reshape(samples1.shape[0], width, width)
            samples2 = samples2.reshape(samples2.shape[0], width, width)

    plt.figure(figsize=(20, 4))
    for i, (sample1, sample2) in enumerate(zip(samples1, samples2)):
        ax = plt.subplot(2, num_samples, i + 1)
        if type == "image":
            plt.imshow(sample1)
            plt.gray()
        else:
            plt.plot(sample1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, num_samples, i + 1 + num_samples)
        if type == "image":
            plt.imshow(sample2)
            plt.gray()
        else:
            plt.plot(sample2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if save:
        plt.savefig(path)
    if plot:
        plt.show()
    # Close the plot to avoid memory leaks
    plt.close()


def evaluate(y_true, y_hat, sel_metric, label="test"):
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
    all_metrics = [metric[0] for metric in all_metrics if not metric[0].startswith("_")]

    # Check if the selected metric is available in sklearn
    if type(sel_metric).__name__ == "function" and sel_metric.__name__ in all_metrics:
        Logger().log_info("Using Scikit-learn metric...")
        result = sel_metric(y_true, y_hat)
        print(
            "{} set results: [\n\t {}: {} \n]".format(
                label, sel_metric.__name__, result
            )
        )
    else:
        Logger().log_info("Using Rapidae custom metric...")
        result = sel_metric.calculate(y_true, y_hat)
        print(
            "{} set results: [\n\t {}: {} \n]".format(
                label, sel_metric.__class__.__name__, result
            )
        )

    return result
