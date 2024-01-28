from typing import Tuple, Union

import keras

from rapidae.conf import Logger
from rapidae.models.base import BaseAE


class RVE(BaseAE):
    """
    Recurrent Variational Encoder (RVE) model.

    Args:
        input_dim (Union[Tuple[int, ...], None]): Shape of the input data.
        latent_dim (int): Dimension of the latent space.
        encoder (BaseEncoder): An instance of BaseEncoder.
        downstream_task (str): Downstream task, can be 'regression' or 'classification'.
        **kwargs (dict): Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim: Union[Tuple[int, ...], None] = None,
        latent_dim: int = 2,
        encoder: callable = None,
        downstream_task: str = None,
        **kwargs,
    ):
        if encoder is None:
            from rapidae.models.base.default_architectures import RecurrentEncoder

            Logger().log_info("Using default encoder")
            encoder = RecurrentEncoder

        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
            **kwargs,
        )

        self.downstream_task = downstream_task

        # Initialize sampling layer
        self.sampling = self.Sampling()

        self.downstream_task = downstream_task.lower()

        if self.downstream_task == "regression":
            from rapidae.models.base import BaseRegressor

            Logger().log_info(
                "Regressor available for the latent space of the autoencoder"
            )
            self.regressor = BaseRegressor()
            self.reg_loss_tracker = keras.metrics.Mean(name="reg_loss")

        elif self.downstream_task == "classification":
            from rapidae.models.base import BaseClassifier

            Logger().log_info(
                "Classificator available for the latent space of the autoencoder"
            )
            self.classifier = BaseClassifier(kwargs["n_classes"])
            self.weight_vae = kwargs["weight_vae"] if "weight_vae" in kwargs else 1.0
            self.weight_clf = kwargs["weight_clf"] if "weight_clf" in kwargs else 1.0
            self.clf_loss_tracker = keras.metrics.Mean(name="clf_loss")

        else:
            Logger().log_warning(
                'The downstream task is not a valid string. Available options: "regression" and "classification"'
            )

        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    # keras model call function
    def call(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling([z_mean, z_log_var])
        outputs = {}
        outputs["z"] = z
        outputs["z_mean"] = z_mean
        outputs["z_log_var"] = z_log_var
        if self.downstream_task == "regression":
            reg_prediction = self.regressor(z)
            outputs["reg"] = reg_prediction
        if self.downstream_task == "classification":
            clf_prediction = self.classifier(z)
            outputs["clf"] = clf_prediction

        return outputs

    class Sampling(keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a sample."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1337)

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = keras.ops.shape(z_mean)[0]
            dim = keras.ops.shape(z_mean)[1]
            # Added seed for reproducibility
            epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)

            return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # KL loss
        kl_loss = -0.5 * (
            1
            + y_pred["z_log_var"]
            - keras.ops.square(y_pred["z_mean"])
            - keras.ops.exp(y_pred["z_log_var"])
        )
        kl_loss = keras.ops.mean(keras.ops.sum(kl_loss, axis=1))
        self.kl_loss_tracker.update_state(kl_loss)

        # Regressor loss
        if self.downstream_task == "regression":
            reg_loss = keras.ops.mean(keras.losses.mean_squared_error(y, y_pred["reg"]))
            self.reg_loss_tracker.update_state(reg_loss)
            loss = kl_loss + reg_loss

        # Classifier loss
        if self.downstream_task == "classification":
            clf_loss = keras.ops.mean(
                keras.losses.categorical_crossentropy(y, y_pred["clf"])
            )
            self.clf_loss_tracker.update_state(clf_loss)
            loss = self.weight_vae * kl_loss + self.weight_clf * clf_loss

        return loss
