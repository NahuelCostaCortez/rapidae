from typing import Optional, Union, Tuple

import keras_core as keras

from aepy.models.base import BaseAE, BaseDecoder, BaseEncoder


class BAG_AE(BaseAE):
    """
    Vanilla Autoencoder model.

    Args:
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
        **kwargs
    ):
        BaseAE.__init__(self, input_dim, latent_dim,
                        encoder=encoder, decoder=decoder, layers_conf=layers_conf)

    # keras model call function
    def call(self, x):
        #sprint(len(inputs))
        #x, y = inputs 
        #x = inputs['data']
        z = self.encoder(x)
        recon_x = self.decoder(z)
        outputs = {}
        outputs["z"] = z
        outputs["recon"] = recon_x
        return outputs
    """
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        print(y.shape)
        loss = keras.ops.sum(keras.losses.mean_squared_error(y, y_pred['recon']))
        return loss
    """

    def train_step(self, *args, **kwargs):
        print("llego")