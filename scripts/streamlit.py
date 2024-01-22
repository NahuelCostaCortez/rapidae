import streamlit as st

import os
import sys
import inspect
import importlib
import numpy as np


current_dir = os.path.abspath("")  # get the current notebook directory
sys.path.append(
    os.path.join(current_dir, "..", "src")
)  # add src folder to path to import modules

from rapidae.data import load_dataset
from rapidae.models import load_model
from rapidae.pipelines import TrainingPipeline
from keras.callbacks import Callback
import plotly.graph_objects as go


def list_datasets():
    # Import the module
    data_module = importlib.import_module("rapidae.data")

    # Get all functions in the module
    functions = inspect.getmembers(data_module, inspect.isfunction)

    # Filter functions that start with 'load_'
    load_functions = [func for func in functions if func[0].startswith("load_")]

    # Return the names of the functions in a list
    dataset_names = [func[0].split("_")[1] for func in load_functions]
    # exclude the load_dataset function
    dataset_names.remove("dataset")
    return dataset_names


def process_data(dataset, data):
    if dataset == "MNIST":
        x_train = data["x_train"].reshape(data["x_train"].shape[0], -1) / 255
        x_test = data["x_test"].reshape(data["x_test"].shape[0], -1) / 255
        data["x_train"] = x_train
        data["x_test"] = x_test
        from keras import utils

        # Obtain number of clasess
        n_classes = len(set(data["y_train"]))

        # Convert labels to categorical
        y_train = utils.to_categorical(data["y_train"], n_classes)
        y_test = utils.to_categorical(data["y_test"], n_classes)
        data["y_train"] = y_train
        data["y_test"] = y_test

    if dataset == "CMAPSS":
        from rapidae.data import PreprocessPipeline, CMAPSS_preprocessor

        preprocess_pipeline = PreprocessPipeline(
            name="CMAPPS_preprocessing", preprocessor=CMAPSS_preprocessor
        )

        x_train, y_train, x_val, y_val, x_test, y_test = preprocess_pipeline(
            train=data["train"], test=data["test"], y_test=data["y_test"], threshold=100
        )

        data["x_train"] = x_train
        data["y_train"] = y_train
        data["x_val"] = x_val
        data["y_val"] = y_val
        data["x_test"] = x_test
        data["y_test"] = y_test

    return data


class StreamlitCallback(Callback):
    def __init__(self):
        super().__init__()
        self.placeholder = st.empty()
        self.fig = go.Figure()

    def on_train_begin(self, logs=None):
        self.loss_values = []
        self.val_loss_values = []

    def on_epoch_end(self, epoch, logs=None):
        with self.placeholder.container():
            loss = logs.get("loss")
            self.loss_values.append(loss)
            self.fig.data = []
            self.fig.add_trace(go.Scatter(y=self.loss_values, mode="lines"))
            self.fig.update_layout(
                title="Loss Plot", xaxis_title="Epoch", yaxis_title="Loss"
            )
            # check if validation loss is available
            if "val_loss" in logs.keys():
                val_loss = logs.get("val_loss")
                self.val_loss_values.append(val_loss)
                self.fig.add_trace(go.Scatter(y=self.val_loss_values, mode="lines"))
            st.write(self.fig)


def main():
    st.set_page_config(layout="wide")

    # Header
    st.title("Rapidae Web Interface")

    # Above part - Data selection and model selection
    left_column, middle_column, right_column = st.columns(3)

    # Left section - Data selection
    with left_column:
        st.subheader("Data")
        # Add your code for data selection here
        # create a selectbox to choose a dataset
        dataset = st.selectbox("Select a dataset", list_datasets(), index=2)

    # Middle section - Model selection
    with middle_column:
        st.subheader("Model")
        # Add your code for model selection here
        model_name = st.selectbox(
            "Select a model",
            [
                "Autoencoder (AE)",
                "Denoising Autoencoder",
                "Sparse Autoencoder",
                "Contractive Autoencoder",
                "Variational Autoencoder (VAE)",
                "Vector Quantised-Variational AutoEncoder (VQ-VAE)",
                "Recurrent Variational Encoding (RVE)",
            ],
        )

    # Right section - Hyperparameter selection
    with right_column:
        st.subheader("Hyperparameters")
        # Add your code for hyperparameter selection here
        # create a selectbox to choose a dataset
        latent_space = st.slider("Latent Space Dimension", 2, 100, value=2)
        epochs = st.slider("Epochs", 1, 100, 5)
        # batch size, these should be only powers of 2
        batch_size = st.selectbox(
            "Batch size", options=np.power(2, np.arange(0, 11)).tolist()
        )
        learning_rate = st.selectbox(
            options=[0.001, 0.01, 0.1, 1.0], label="Learning Rate"
        )

    # Add a button to start training
    train = st.button("Train")

    # Below part - Model training execution and results
    st.header("Model Training")
    # Add your code for model training execution and results here
    if train:
        from streamlit_tensorboard import st_tensorboard
        from tensorboard import program
        from keras.callbacks import TensorBoard, ModelCheckpoint

        data = load_dataset(dataset)
        data = process_data(dataset, data)
        model = load_model(model_name, data["x_train"].shape[1], latent_space)

        log_dir = "logs"

        pipeline = TrainingPipeline(
            name="training_pipeline_" + model_name,
            model=model,
            num_epochs=epochs,
            batch_size=batch_size,
            output_dir=log_dir,
        )

        pipeline.callbacks = [StreamlitCallback()]

        if "x_val" in data.keys():
            pipeline.callbacks.append(
                ModelCheckpoint(
                    filepath=log_dir + "/model.weights.h5",
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True,
                    mode="min",
                    save_weights_only=True,
                )
            )
            trained_model = pipeline(
                x=data["x_train"],
                y=data["y_train"],
                x_eval=data["x_val"],
                y_eval=data["y_val"],
            )
        else:
            pipeline.callbacks.append(
                ModelCheckpoint(
                    filepath=log_dir + "/model.weights.h5",
                    monitor="loss",
                    verbose=1,
                    save_best_only=True,
                    mode="min",
                    save_weights_only=True,
                )
            )
            trained_model = pipeline(
                x=data["x_train"],
                y=data["y_train"],
            )

        # Start TensorBoard
        # tb = program.TensorBoard()
        # tb.configure(argv=[None, "--logdir", log_dir])
        # url = tb.launch()
        # st_tensorboard(url)


if __name__ == "__main__":
    main()
