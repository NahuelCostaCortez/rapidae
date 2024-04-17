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
        Logger().log_info("Saving latent space plot...")
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
        Logger().log_info("Saving reconstructions plot...")
        plt.savefig(path)
    if plot:
        plt.show()
    # Close the plot to avoid memory leaks
    plt.close()


def plot_samplings_from_latent(model, n=30, figsize=15):
    """
    Generate and display a 2D manifold of decoded samples from the latent space.

    This function generates a 2D grid in the latent space and decodes each point to visualize the distribution
    of digit classes in the latent space. The resulting grid is displayed as a grayscale image.

    Args:
        model (keras.Model): A trained latent model.
        n (int, optional): The number of points along each axis in the 2D manifold grid. 
        figsize (int, optional): The size of the generated matplotlib figure. 

    Returns:
        None
    """
    # display a n*n 2D manifold 
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decoder(z_sample)
            digit = (np.array(x_decoded)).reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


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
