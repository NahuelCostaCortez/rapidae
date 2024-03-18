from typing import Union, Tuple
from keras import ops, losses
from rapidae.models.base import BaseAE


class AE(BaseAE):
    """
    Vanilla Autoencoder model.

    Args:
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        encoder (BaseEncoder): An instance of BaseEncoder.
        decoder (BaseDecoder): An instance of BaseDecoder.
        layers_conf (list): List specifying the configuration of layers for custom models. Only applies for MLP models.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
        **kwargs,
    ):
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            decoder=decoder,
            exclude_decoder=False,
            layers_conf=layers_conf,
            **kwargs,
        )

    def call(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        outputs = {}
        outputs["z"] = z
        outputs["x_recon"] = recon_x

        return outputs

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        loss = ops.mean(losses.mean_squared_error(x, y_pred["x_recon"]))

        return loss
