from keras import Sequential, ops, losses, metrics
from keras.layers import GlobalAveragePooling1D, Dense, Dropout
from rapidae.models.base import BaseAE

class ICFormer(BaseAE):
    def __init__(
        self,
        input_dim=12,
        latent_dim=128,
        mlp_units=128,
        mlp_dropout=0.0,
        encoder = None,
    ):
        BaseAE.__init__(
            self,
            input_dim,
            latent_dim,
            encoder=encoder,
        )

        self.mlp_layer = Sequential(
            [
                GlobalAveragePooling1D(data_format="channels_first"),
                Dense(mlp_units, activation="relu"),
            ]
        )

        self.dropout = Dropout(mlp_dropout)

        self.regression_layer = Dense(
            3 * input_dim, activation="linear", name="regression_output"
        )
        self.classification_layer = Dense(
            1, activation="sigmoid", name="classification_output"
        )

        self.loss_tracker = metrics.Mean(name="loss")
        self.regression_tracker = metrics.Mean(name="reg_loss")
        self.classification_tracker = metrics.Mean(name="clf_loss")

    def call(self, x, training=True):
        encoder_outputs, attention_outputs, attention_weights, attention_outputs_sum = (
            self.encoder(x, training=training, mask=None)
        )

        mlp_output = self.mlp_layer(encoder_outputs)
        mlp_output = self.dropout(mlp_output, training=training)

        regression_output = self.regression_layer(mlp_output)
        classification_output = self.classification_layer(
            mlp_output
        )

        return {
            "regression_output": regression_output,
            "classification_output": classification_output,
            "attention_outputs": attention_outputs,
            "attention_weights": attention_weights,
            "attention_outputs_sum": attention_outputs_sum,
        }

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        loss_reg = losses.mean_squared_error(
            y["y_regression"], y_pred["regression_output"]
        )
        self.regression_tracker.update_state(loss_reg)

        loss_clf = losses.binary_crossentropy(
            y["y_classification"], y_pred["classification_output"]
        )
        self.classification_tracker.update_state(loss_clf)

        loss_ = loss_reg + loss_clf * 10
        self.loss_tracker.update_state(loss_)

        return ops.mean(loss_)
