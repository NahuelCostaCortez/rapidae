"""
Class for common model utilities.
"""


def list_models():
    """
    List all available models.
    Note that these are the base models, for a complete guide of the implemented models, please refer to the documentation.
    """

    import os
    import pkg_resources

    # Get the path to the 'models' directory
    directory = pkg_resources.resource_filename("rapidae", "models")

    # Get the names of the folders in the directory
    folders = [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and name != "__pycache__"
    ]

    # return the list of models in uppercase
    return [model.upper() for model in folders]


def load_model(model_name, input_dim, latent_dim, **kwargs):
    """
    Load a model.

    Args:
        model_name (str): Name of the model to load.
    """
    import importlib

    try:
        # convert model name to lowercase
        model_name = model_name.lower()

        # Dynamically import the model module
        model_module = importlib.import_module(
            "." + model_name, package="rapidae.models"
        )

        # convert model name to uppercase
        model_name = model_name.upper()

        # Get the model class from the module
        model_class = getattr(model_module, model_name)

        # Create an instance of the model class
        model = model_class(input_dim, latent_dim, **kwargs)

        return model

    except ImportError:
        raise ValueError(f"Model '{model_name}' not found.")
