"""
This module contains everything related to the acquisition and preprocessing of data sets. It also includes a tool utils for various tasks: latent space visualization, evaluation, etc.
"""

from .datasets import load_CMAPSS, load_MNIST, load_arrhythmia_data
from .preprocessing import CMAPSS_preprocessor
from .utils import plot_label_clusters, evaluate, add_noise, display_diff

__all__ = [
    "load_CMAPSS",
    "load_MNIST",
    "load_arrhythmia_data",
    "CMAPSS_preprocessor",
    "plot_label_clusters",
    "evaluate",
    "add_noise",
    "display_diff",
]
