from typing import Optional, Tuple, Union

import tensorflow as tf

from models.base import BaseAE, BaseDecoder, BaseEncoder, BaseRegressor


class VAE(BaseAE):
    """
    Variational Autoencoder (VAE) model.

    Args:
        model_config (BaseAEConfig): configuration object for the model
        encoder (BaseEncoder): An instance of BaseEncoder. Default: None
        decoder (BaseDecoder): An instance of BaseDecoder. Default: None
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        masking_value: float = -99.0,
        exclude_decoder: bool = False,
        regressor: tf.keras.layers.Layer = None,
        encoder: callable = None,
        decoder: callable = None,
        layers_conf: list = None,
    ):
        
        BaseAE.__init__(self, input_dim, latent_dim,
                        encoder=encoder, decoder=decoder, masking_value=masking_value, layers_conf=layers_conf)
        if regressor is not False:
            self.regressor = BaseRegressor()
            self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        else:
            self.regressor = False
        

        if self.decoder is not False:
            #self.decoder = decoder
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
                name="reconstruction_loss")

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    # keras model call function
    @tf.function
    def call(self, inputs):
        x = inputs["data"]
        z_mean, z_log_var = self.encoder(x)
        z = self.Sampling()([z_mean, z_log_var])
        outputs = {}
        outputs["z"] = z
        outputs["z_mean"] = z_mean
        outputs["z_log_var"] = z_log_var
        if self.regressor != None:
            reg_prediction = self.regressor(z)
            outputs["reg"] = reg_prediction
        if self.decoder != None:
            recon_x = self.decoder(z)
            outputs["recon"] = recon_x
        return outputs

    # TO-DO: cambiar a una función que se llame NormalSampler
    class Sampling(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a sample."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            # added seed for reproducibility
            epsilon = tf.random.normal(shape=(batch, dim), seed=42)
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @property
    def metrics(self):
        metrics = [self.total_loss_tracker, self.kl_loss_tracker]
        if self.decoder != None:
            metrics.append(self.reconstruction_loss_tracker)
        #if self.regressor != None:
        #    metrics.append(self.reg_loss_tracker)
        return metrics

    @tf.function
    def train_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]
        with tf.GradientTape() as tape:
            metrics = {}

            # kl loss
            z_mean, z_log_var = self.encoder(x)
            z = self.Sampling()([z_mean, z_log_var])
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            self.kl_loss_tracker.update_state(kl_loss)
            metrics["kl_loss"] = kl_loss

            total_loss = kl_loss

            # Regressor
            if self.regressor != None:
                reg_prediction = self.regressor(z)
                reg_loss = tf.reduce_mean(
                    tf.keras.losses.mse(labels_x, reg_prediction)
                )
                self.reg_loss_tracker.update_state(reg_loss)
                metrics["reg_loss"] = reg_loss

                total_loss += reg_loss

            # Reconstruction
            if self.decoder != None:
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.mse(x, reconstruction)
                )
                print(self.regressor)
                if self.regressor is not False:
                    total_loss = kl_loss + reg_loss + reconstruction_loss
                else:
                    total_loss = kl_loss + reconstruction_loss
                self.reconstruction_loss_tracker.update_state(
                    reconstruction_loss)
                metrics["reconstruction_loss"] = reconstruction_loss
                total_loss += reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        metrics["loss"] = total_loss

        return metrics

    @tf.function
    def test_step(self, inputs):
        x = inputs["data"]
        labels_x = inputs["labels"]

        metrics = {}

        # kl loss
        z_mean, z_log_var = self.encoder(x)
        z = self.Sampling()([z_mean, z_log_var])
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        metrics["kl_loss"] = kl_loss

        total_loss = kl_loss

        # Regressor
        if self.regressor != None:
            reg_prediction = self.regressor(z)
            reg_loss = tf.reduce_mean(
                tf.keras.losses.mse(labels_x, reg_prediction)
            )
            metrics["reg_loss"] = reg_loss
            total_loss += reg_loss

        # Reconstruction
        if self.decoder != None:
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(x, reconstruction)
            )
            metrics["reconstruction_loss"] = reconstruction_loss
            total_loss += reconstruction_loss

        metrics["loss"] = total_loss

        return metrics
