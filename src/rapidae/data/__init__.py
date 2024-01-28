"""
This module contains everything related to the acquisition and preprocessing of data sets. It also includes a tool utils for various tasks: latent space visualization, evaluation, etc.
"""

from .datasets import (
    list_datasets,
    load_dataset,
    load_CMAPSS,
    load_MNIST,
    load_AtrialFibrillation,
)
from .preprocessing import CMAPSS_preprocessor
from .utils import add_noise

__all__ = [
    "list_datasets",
    "load_dataset",
    "load_CMAPSS",
    "load_MNIST",
    "load_AtrialFibrillation",
    "CMAPSS_preprocessor",
    "add_noise",
]
