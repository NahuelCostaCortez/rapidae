"""
This module serves as the foundation for handling data. It includes several benchmark datasets and encompasses functionalities for data acquisition and preprocessing. The organization of this module ensures that users can easily manage and preprocess predefined or custom data, streamlining the model preparation process.
"""

from .datasets import (
    list_datasets,
    load_dataset,
    load_CMAPSS,
    load_MNIST,
    load_AtrialFibrillation,
)
from .preprocessing import CMAPSS_preprocessor
from .utils import *

__all__ = [
    "list_datasets",
    "load_dataset",
    "load_CMAPSS",
    "load_MNIST",
    "load_AtrialFibrillation",
    "CMAPSS_preprocessor",
]
