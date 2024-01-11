from typing import Union, Tuple

import keras

from rapidae.models.base import BaseAE


class CAE(BaseAE):
    """
    Contractive Autoencoder model.

    Args:
        encoder (BaseEncoder): An instance of BaseEncoder. 
        decoder (BaseDecoder): An instance of BaseDecoder.
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data. 
        latent_dim (int): Dimension of the latent space.
        layers_conf (list): List specifying the configuration of layers for custom models. 
        lambda_ (float): Weight of the contractive loss term. 
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            input_dim: Union[Tuple[int, ...], None] = None,
            latent_dim: int = 2,
            encoder: callable = None,
            decoder: callable = None,
            layers_conf: list = None,
            lambda_ = 1e-4,
            **kwargs
    ):
        BaseAE.__init__(self, input_dim, latent_dim,
                        encoder=encoder, decoder=decoder, layers_conf=layers_conf, **kwargs)
        
        self.lambda_ = lambda_
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name='reconstruction_loss')
        self.contractive_loss_tracker = keras.metrics.Mean(
            name='contractive_loss')


    def call(self, x):
        x_hid = self.encoder(x)
        recon_x = self.decoder(x_hid)
        outputs = {}
        outputs['x_hidden'] = x_hid
        outputs['recon'] = recon_x

        return outputs
    
    
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # Reconstruction loss
        recon_loss = keras.ops.mean(
            keras.losses.mean_squared_error(x, y_pred['recon'])
        )
        self.reconstruction_loss_tracker.update_state(recon_loss)

        # Contractive loss
        last_layer = self.encoder.enc_layer
        W = last_layer.weights[0]  # N x N_hidden
        W = keras.ops.transpose(W)  # N_hidden x N
        h = y_pred['x_hidden']
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = self.lambda_ * \
            keras.ops.sum(keras.ops.matmul(dh**2, keras.ops.square(W)), axis=1)
        self.contractive_loss_tracker.update_state(contractive_loss)

        # Total loss
        loss = contractive_loss + recon_loss

        return loss