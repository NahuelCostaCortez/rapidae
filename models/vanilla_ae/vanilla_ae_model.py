from typing import Optional

import tensorflow as tf

from models.base import BaseAE, BaseDecoder, BaseEncoder

from .vanilla_ae_config import VanillaAEConfig


class VanillaAE(BaseAE):
    """
    Vanilla Autoencoder (AE) model.
    
    Args:
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """

    def __init__(
        self,
        model_config: VanillaAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        if self.decoder is not False:
            self.decoder = decoder
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    # keras model call function
    @tf.function
    def call(self, inputs):
        x = inputs['data']
        encoded = self.encoder(x)
        outputs = {}
        if self.decoder!=None:
            recon_x = self.decoder(encoded)
            outputs["recon"] = recon_x
        return outputs

    @property
    def metrics(self):
        metrics = [self.total_loss_tracker]
        if self.decoder!=None:
            metrics.append(self.reconstruction_loss_tracker)
        return metrics

    @tf.function
    def train_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]
        with tf.GradientTape() as tape:
            metrics = {}
        return 0

    @tf.function
    def test_step(self, inputs):
        return 0
        

        