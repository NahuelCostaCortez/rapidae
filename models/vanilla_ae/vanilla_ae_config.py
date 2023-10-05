from pydantic.dataclasses import dataclass

from models.base import BaseAEConfig


@dataclass
class VanillaAEConfig(BaseAEConfig):
    """
    AE config class.
    
    Parameters:
	    - input_dim (tuple): The input_data dimension in the form (seq_len, num_features). Default: None.
        - latent_dim (int): The latent space dimension. Default: 2.
        - masking_value (float): The masking value for those values that must be ommited. Default: -99.0.
	    - decoder (bool): Whether to include the decoder. Default: True, set to false if you want to exclude it from traning.
		- regressor (bool): Whether to include a regressor. Default: False.
    """
    exclude_decoder: bool = False
