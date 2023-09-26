from .ae_config import AEConfig
from typing import Optional
from models.base import BaseAE, BaseEncoder, BaseDecoder
import tensorflow as tf

class AE(BaseAE):
    """
    Vanilla Autoencoder model.
    
    Args:
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """
    
    def __init__(
        self,
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
    
        BaseAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
    
        self.decoder = decoder
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
    
    # keras model call function
    def call(self, inputs):
        x = inputs["data"]
        z = self.encoder(x)
        recon_x = self.decoder(z)
        outputs = {}
        outputs["z"] = z
        outputs["recon"] = recon_x
        return outputs
    
    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker]
    
    @tf.function
    def train_step(self, inputs):
        x = inputs["data"]
        with tf.GradientTape() as tape:
            metrics = {}
    
            z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(x, reconstruction)
            )
            metrics["reconstruction_loss"] = reconstruction_loss

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        
        return metrics
    
    @tf.function
    def test_step(self, inputs):
        x = inputs[0]["data"]
        labels_x = inputs[0]["labels"]

        metrics = {}

        z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x, reconstruction)
        )
        metrics["reconstruction_loss"] = reconstruction_loss

        return metrics
        
        
    
        