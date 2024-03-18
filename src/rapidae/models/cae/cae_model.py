from typing import Union, Tuple
from keras import metrics, ops, losses
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
        **kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
        lambda_=1e-4,
        **kwargs,
    ):
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            decoder=decoder,
            layers_conf=layers_conf,
            **kwargs,
        )

        self.lambda_ = lambda_
        # Training metrics trackers
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.contractive_loss_tracker = metrics.Mean(name="contractive_loss")

    def call(self, x):
        x_hid = self.encoder(x)
        recon_x = self.decoder(x_hid)

        return {"x_hidden": x_hid, "x_recon": recon_x}

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # Reconstruction loss
        recon_loss = ops.mean(losses.mean_squared_error(x, y_pred["x_recon"]))
        self.reconstruction_loss_tracker.update_state(recon_loss)

        # Contractive loss
        last_layer = self.encoder.enc_layer
        W = last_layer.weights[0]  # N x N_hidden
        W = ops.transpose(W)  # N_hidden x N
        h = y_pred["x_hidden"]
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive_loss = self.lambda_ * ops.sum(
            ops.matmul(dh**2, ops.square(W)), axis=1
        )
        self.contractive_loss_tracker.update_state(contractive_loss)

        # Total loss
        loss = contractive_loss + recon_loss

        return loss
