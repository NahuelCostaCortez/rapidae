from typing import Tuple, Union

from pydantic.dataclasses import dataclass


@dataclass
class BaseAEConfig():
    """
    This is the base configuration instance of the models`.

    Parameters
    ----------
        input_dim (tuple): The input_data dimension. Default: None.
        latent_dim (int): The latent space dimension. Default: 2.
    """

    # it can be either a tuple of integers or None
    input_dim: Union[Tuple[int, ...], None] = None
    latent_dim: int = 2
    uses_default_encoder: bool = True
    uses_default_decoder: bool = True
