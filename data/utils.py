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
        sel_metric (callable): Class name of the selected metric.
        label (str): Tag of the evaluated set.
    """
    logger = Logger()

    # Get all functions from the metrics module
    all_metrics = getmembers(metrics, isfunction)

    # Filter out private functions
    all_metrics = [metric[0] for metric in all_metrics if not metric[0].startswith('_')]

    # Check if the selected metric is avavilable in sklearn
    if sel_metric in all_metrics:
        result = sel_metric(y_true, y_hat)
    else:
        try:
            result = sel_metric.calculate(y_true, y_hat)
        except:
            logger.log_error('Metric is not available in AEPY neither in Scikit-Learn.')

    print('{} set results: [\n\t {}: {} \n]'.format(
        label, sel_metric.__class__.__name__, result))


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
