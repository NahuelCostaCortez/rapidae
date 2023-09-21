from aepy.models.base import BaseAEConfig
from pydantic.dataclasses import dataclass

@dataclass
class AEConfig(BaseAEConfig):
    """
    AE config class.
    
    Parameters:
	    - input_dim (tuple): The input_data dimension in the form (seq_len, num_features). Default: None.
      - latent_dim (int): The latent space dimension. Default: 2.
    """