"""
This is the core module of the library. It includes the base architectures on which new ones can be created, several predefined architectures and a list of predefined default encoders and decoders.

We will keep updating!
"""

from .ae import AE
from .base import BaseAE, BaseEncoder, BaseDecoder, BaseRegressor, BaseClassifier
from .cae import CAE
from .vae import VAE
from .vq_vae import VQ_VAE, VectorQuantizer
from .utils import LRFinder, list_models

__all__ = [
    "AE",
    "BaseAE",
    "BaseEncoder",
    "BaseDecoder",
    "BaseRegressor",
    "BaseClassifier",
    "CAE",
    "VAE",
    "VQVAE",
    "VectorQuantizer",
    "LRFinder",
    "list_models",
]
