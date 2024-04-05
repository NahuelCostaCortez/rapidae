import os
from datetime import datetime
from typing import Optional

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.backend import backend

from rapidae.models.base import BaseAE
from rapidae.pipelines.base import BasePipeline


class TrainingPipeline(BasePipeline):
    """
    TrainingPipeline class for training autoencoder models.

    This pipeline is responsible for training an autoencoder model using specified parameters.
    It extends the BasePipeline class and includes functionality for model training, evaluation,
    and saving the best weights.

    Attributes:
        name (str): Name of the pipeline.
        model (BaseAE): Model to be trained.
        output_dir (str): Output directory for saving the trained model. If not specified, a new directory "output_dir/name_YYY-MM-DD_HH-MM-SS" is created. Defaults to "output_dir".
        optimizer (str): Name of the optimizer. Currently only 'adam' is supported.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        callbacks (list, optional): List of Keras callbacks. If None, EarlyStopping and ModelCheckpoint are created. Defaults to None.
        save_model (bool, optional): Flag to save the trained model. Defaults to True. Otherwise, the output directory is removed.
        run_eagerly (bool, optional): Flag to run the model eagerly. Defaults to False.
        verbose (int, optional): Verbosity mode. 0 will show no output, 1 will show a progress bar, and 2 will just mention the number of epoch. Defaults to 2.
    """

    def __init__(
        self,
        name=None,
        model: Optional[BaseAE] = None,
        output_dir="output_dir",
        optimizer="adam",
        learning_rate=0.001,
        batch_size=128,
        num_epochs=100,
        callbacks: Optional[list] = None,
        save_model: bool = True,
        run_eagerly=False,
        verbose=2,
    ):
        super().__init__(name, output_dir)
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.callbacks = callbacks
        self.save_model = save_model
        self.run_eagerly = run_eagerly
        self.verbose = verbose

    def plot_training_history(self):
        """
        Plots the training history of the model.
        """
        import matplotlib.pyplot as plt

        # check first what metrics are available
        for key in self.model.history.history.keys():
            plt.plot(self.model.history.history[key], label=key)
        plt.title("Traning history")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    def handle_callbacks(self, x_eval):
        """
        Handles the callbacks for the training pipeline.
        """

        has_early_stopping = False
        has_model_checkpoint = False
        monitor = None

        # Check if there are callbacks for early stopping and model checkpoint
        if self.callbacks is None:
            self.callbacks = []
        else:
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    has_early_stopping = True
                if isinstance(callback, ModelCheckpoint):
                    has_model_checkpoint = True
                # check if callback has attribute path
                if hasattr(callback, "path"):
                    # set the path to the output_dir
                    self.logger.log_info("Setting path to output_dir")
                    callback.path = os.path.join(self.output_dir, callback.path)
                    # create path if it does not exist
                    if not os.path.exists(callback.path):
                        os.makedirs(callback.path)

        # Check if validation data is provided
        monitor = "val_loss" if x_eval is not None else "loss"

        # If early stopping is not provided, add it
        if not has_early_stopping:
            self.callbacks.append(
                EarlyStopping(monitor=monitor, patience=10, verbose=1, mode="min")
            )

        # If model checkpoint is not provided, add it
        if not has_model_checkpoint:
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.output_dir, "model.weights.h5"),
                    # filepath=os.path.join(self.output_dir, "model.keras"),
                    monitor=monitor,
                    verbose=1,
                    save_best_only=True,
                    mode="min",
                    # save_weights_only=False,
                    save_weights_only=True,
                )
            )

    def __call__(
        self,
        x,
        y=None,
        x_eval=None,
        y_eval=None,
    ):
        """
        Launches the training pipeline.

        Args:
            x (ArrayLike): Training input data.
            y (ArrayLike, optional): Training target data. Defaults to None.
            x_eval (ArrayLike, optional): Validation input data. Defaults to None.
            y_eval (ArrayLike, optional): Validation target data. Defaults to None.

        Returns:
            BaseAE (keras.Model): Trained model.
        """

        # ------------ Create output directory ------------
        if self.output_dir == "output_dir":
            # create a dir self.output_dir/training_YYY-MM-DD_HH-MM-SS
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_path = os.path.join(".", self.output_dir, f"{self.name}_{timestamp}")
        else:
            folder_path = self.output_dir

        self.logger.log_info("+++ {} +++".format(self.name))
        self.logger.log_info("Creating folder in {}".format(folder_path))

        # if folder does not exist create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.output_dir = str(folder_path)
        # ------------------------------------------------

        # --------------- Handle callbacks ---------------
        self.handle_callbacks(x_eval)
        # ------------------------------------------------

        # ----------------- Compile model ----------------
        if self.optimizer == "adam":
            optimizer = Adam(learning_rate=self.learning_rate)
        else:
            self.logger.log_error("Unimplemented optimizer")
            exit(-1)

        self.model.compile(optimizer=optimizer, run_eagerly=self.run_eagerly)
        # ------------------------------------------------
        validation_data = (
            (x_eval, y_eval) if x_eval is not None and y_eval is not None else None
        )
        self.logger.log_info(
            "\nTRAINING STARTED\n\tBackend: {}\n\tEager mode: {}\n\tValidation data available: {}\n\tCallbacks set: {} \n".format(
                backend(),
                self.run_eagerly,
                validation_data is not None,
                [callback.__class__.__name__ for callback in self.callbacks],
            )
        )

        self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            shuffle=True,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )

        # Restore the best model
        # check if file exists
        if not os.path.exists(os.path.join(self.output_dir, "model.weights.h5")):
            self.logger.log_error("Best model not found")
        else:
            self.logger.log_info("Restoring best model")
            self.model.load_weights(os.path.join(self.output_dir, "model.weights.h5"))
            self.logger.log_info("Best model restored")

        if self.save_model is False:
            os.remove(self.output_dir)

        return self.model
