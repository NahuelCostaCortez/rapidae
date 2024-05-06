"""
Class to load some common datasets.
"""

import os

import urllib
import numpy as np
import pandas as pd
import keras
from shutil import rmtree
from rapidae.data import utils

from rapidae.conf import Logger


def load_SineWave(persistant=False):
    """
    Load SineWave dataset.

    Args:
        persistant (bool): If True, keeps the downloaded dataset files.
                           If False, deletes the dataset files after loading.
                           Default is False.

    Returns:
        data (dict): Dictionary containing the data.
    """
    # Load the data
    data_dir = os.path.join(".", "datasets", "SineWave")
    data = utils.get_data_from_url(
        url="https://raw.githubusercontent.com/NahuelCostaCortez/datasets/main/SineWave/sine_wave.npy",
        filename="sine_wave.npy",
        data_dir=data_dir,
        persistant=persistant,
    )

    return data


def load_AtrialFibrillation(persistant=False):
    """
    Load AtrialFibrillation dataset.

    Args:
        persistant (bool): If True, keeps the downloaded dataset files.
                           If False, deletes the dataset files after loading.
                           Default is False.

    Returns:
        data (dict): Dictionary containing the data.
    """
    # Load the data
    data_dir = os.path.join(".", "datasets", "AtrialFibrilation")
    data = utils.get_data_from_url(
        url="https://raw.githubusercontent.com/NahuelCostaCortez/datasets/main/AtrialFibrillation/arrhythmia_data.npy",
        filename="arrhythmia_data.npy",
        data_dir=data_dir,
        persistant=persistant,
    )

    return data


def load_MNIST(persistant=False):
    """
    Returns data for the MNIST dataset.
    It can be obtained from original source or from Keras repository.

    Args:
        persistant (bool): If True, keeps the downloaded dataset files.
                           If False, deletes the dataset files after loading.
                           Default is False.

    Returns:
        data (dict): Dictionary containing the data.
    """

    Logger().log_info(f"Downloading data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train =  np.expand_dims(x_train, -1).astype("float32")
    x_test = np.expand_dims(x_test, -1).astype("float32")

    data_dir = os.path.join(".", "datasets", "MNIST")
    # If required delete data
    if persistant:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
            np.save(os.path.join(data_dir, "x_train.npy"), x_train)
            np.save(os.path.join(data_dir, "y_train.npy"), y_train)
            np.save(os.path.join(data_dir, "x_test.npy"), x_test)
            np.save(os.path.join(data_dir, "y_test.npy"), y_test)
        else:
            Logger().log_info(f"Skipping... Data already exists.")
    else:
        if os.path.exists(data_dir):
            Logger().log_info("Deleting MNIST data...")
            rmtree(data_dir)
            # check if datasets folder is empty
            if not os.listdir("./datasets"):
                os.rmdir("./datasets")

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    '''
    url_base = "http://yann.lecun.com/exdb/mnist/"
    filenames = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    data_dir = os.path.join(".", "datasets", "MNIST")

    train_img_path = os.path.join(data_dir, filenames[0])
    train_lbl_path = os.path.join(data_dir, filenames[1])
    test_img_path = os.path.join(data_dir, filenames[2])
    test_lbl_path = os.path.join(data_dir, filenames[3])

    # Create a directory to store the downloaded files
    os.makedirs(data_dir, exist_ok=True)

    # Download MNIST dataset files
    for filename in filenames:
        url = url_base + filename
        target_path = os.path.join(data_dir, filename)
        if not os.path.exists(target_path):
            Logger().log_info(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, target_path)
        else:
            Logger().log_info(f"{filename} already exists.")

    # Load the training and test data
    x_train = utils.mnist_load_images(train_img_path)
    y_train = utils.mnist_load_labels(train_lbl_path)
    x_test = utils.mnist_load_images(test_img_path)
    y_test = utils.mnist_load_labels(test_lbl_path)

    # If required delete data
    if not persistant:
        Logger().log_info("Deleting MNIST data...")
        rmtree(data_dir)
        # check if datasets folder is empty
        if not os.listdir("./datasets"):
            os.rmdir("./datasets")

    data = {}
    data["x_train"] = x_train
    data["y_train"] = y_train
    data["x_test"] = x_test
    data["y_test"] = y_test

    return data
    '''


def load_CMAPSS(subset="FD001", persistant=False):
    """
    Returns data for the requested subset of the CMAPSS dataset.

    These are download from the NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational repository
    since they are not longer available in the original source:
    https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq

    Args:
        subset (str): Selected subset of CMAPSS dataset. There are 4 available: FD001, FD002, FD003, FD004
        persistant (bool): If True, keeps the downloaded dataset files.
                           If False, deletes the dataset files after loading.
                           Default is False.

    Returns:
        data (dict): Dictionary containing the data for the selected subset of CMAPSS dataset.
    """

    if subset not in ["FD001", "FD002", "FD003", "FD004"]:
        raise ValueError(
            "Invalid subset. Supported subsets are: FD001, FD002, FD003, FD004"
        )

    # Load the data
    url_train = (
        "https://raw.githubusercontent.com/NahuelCostaCortez/datasets/main/CMAPSS/train_"
        + subset
        + ".txt"
    )
    url_RUL = (
        "https://raw.githubusercontent.com/NahuelCostaCortez/datasets/main/CMAPSS/RUL_"
        + subset
        + ".txt"
    )
    url_test = (
        "https://raw.githubusercontent.com/NahuelCostaCortez/datasets/main/CMAPSS/test_"
        + subset
        + ".txt"
    )

    filenames = [
        "train_" + subset + ".txt",
        "RUL_" + subset + ".txt",
        "test_" + subset + ".txt",
    ]
    data_dir = os.path.join(".", "datasets", "CMAPSS")

    # columns
    index_names = ["unit_nr", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = ["s_{}".format(i + 1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv(url_train, sep=r"\s+", header=None, names=col_names)
    test = pd.read_csv(url_test, sep=r"\s+", header=None, names=col_names)
    y_test = pd.read_csv(
        url_RUL, sep=r"\s+", header=None, names=["RemainingUsefulLife"]
    )

    if persistant:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            Logger().log_info(f"Downloading data...")
            train.to_csv(os.path.join(data_dir, filenames[0]), index=False)
            test.to_csv(os.path.join(data_dir, filenames[2]), index=False)
            y_test.to_csv(os.path.join(data_dir, filenames[1]), index=False)
        else:
            Logger().log_info(f"Skipping... Data already exists.")

    data = {}
    data["x_train"] = train
    data["x_test"] = test
    data["y_test"] = y_test

    return data


def load_dataset(dataset, subset="FD004", persistant=False):
    """
    Load the dataset.

    Args:
        dataset (str): Name of the dataset to be loaded.
                    If False, returns the raw data.
                    Supported datasets are: MNIST, AtrialFibrillation, CMAPSS, SineWave.
        persistant (bool): If True, keeps the downloaded dataset files.
                           If False, deletes the dataset files after loading.
                           Default is False.
    """
    if dataset == "MNIST":
        return load_MNIST(persistant)
    elif dataset == "AtrialFibrillation":
        return load_AtrialFibrillation(persistant)
    elif dataset == "CMAPSS":
        return load_CMAPSS(subset=subset, persistant=persistant)
    elif dataset == "SineWave":
        return load_SineWave(persistant)
    else:
        raise ValueError("Invalid dataset name.")


def list_datasets():
    import inspect
    import sys

    # Import the module
    current_module = sys.modules[__name__]
    # Get all functions in the module
    functions = inspect.getmembers(current_module, inspect.isfunction)
    # Filter functions that start with 'load_'
    load_functions = [func for func in functions if func[0].startswith("load_")]
    # Return the names of the functions in a list
    dataset_names = [func[0].split("_")[1] for func in load_functions]
    # Exclude the load_dataset function
    dataset_names.remove("dataset")
    # Remove duplicates
    dataset_names = list(set(dataset_names))
    # Order alphabetically
    dataset_names.sort()
    return dataset_names
