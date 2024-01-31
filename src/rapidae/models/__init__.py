"""
This is the core module of the library. It includes the base architectures on which new ones can be created, several predefined architectures and a list of predefined default encoders and decoders.

We will keep updating!
"""

from .ae import AE
from .cae import CAE
from .vae import VAE
from .rve import RVE
from .vq_vae import VQ_VAE, VectorQuantizer
from .utils import *


__all__ = [
    "AE",
    "CAE",
    "VAE",
    "VQ_VAE",
    "VectorQuantizer",
    "RVE",
]
