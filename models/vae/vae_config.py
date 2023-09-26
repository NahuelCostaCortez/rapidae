from models.base import BaseAEConfig
from pydantic.dataclasses import dataclass

@dataclass
class VAEConfig(BaseAEConfig):
    """
    VAE config class.
    
    Parameters:
	    - input_dim (tuple): The input_data dimension in the form (seq_len, num_features). Default: None.
        - latent_dim (int): The latent space dimension. Default: 2.
        - masking_value (float): The masking value for those values that must be ommited. Default: -99.0.
	    - decoder (bool): Whether to include the decoder. Default: True, set to false if you want to exclude it from traning.
		- regressor (bool): Whether to include a regressor. Default: False.
    """
    
    masking_value: float = -99.0
    exclude_decoder: bool = False
    regressor: bool = False