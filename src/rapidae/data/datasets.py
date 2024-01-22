"""
Class to load some common datasets.
"""

import gzip
import os
import urllib
from shutil import rmtree

import numpy as np
import pandas as pd
import requests

from rapidae.conf import Logger


def get_data_from_url(url):
    """
    Download data from a specific url.

    Args:
        url (str): Given url where the data will be downloaded.
    """
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text

        return data
    else:
        print("Failed to retrieve data from", url)

        return None


def mnist_load_images(filename):
    """
    Auxiliary function to load gzipped images for MNIST dataset.

    Args:
        filename (str): Path to the file.
    """
    with gzip.open(filename, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)

    return data.reshape(-1, 28, 28, 1)


def mnist_load_labels(filename):
    """
    Auxiliary function to load gzipped labels for MNIST dataset.

    Args:
        filename (str): Path to the file.
    """
    with gzip.open(filename, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)

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
        data (dict): Dictionary containing the data for the MNIST dataset.
    """
    url_base = "http://yann.lecun.com/exdb/mnist/"
    filenames = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    data_dir = os.path.join("..", "datasets", "MNIST")

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
    x_train = mnist_load_images(train_img_path)
    y_train = mnist_load_labels(train_lbl_path)
    x_test = mnist_load_images(test_img_path)
    y_test = mnist_load_labels(test_lbl_path)

    # If required delete data
    if not persistant:
        Logger().log_info("Deleting MNIST data...")
        rmtree(data_dir)

    data = {}
    data["x_train"] = x_train
    data["y_train"] = y_train
    data["x_test"] = x_test
    data["y_test"] = y_test

    return data


def convert_one_hot(x, target):
    """
    Convert target values to one-hot encoding.

    Args:
        x (numpy.ndarray): Input array or matrix.
        target (numpy.ndarray): Target values to be converted to one-hot encoding.

    Returns:
        samples (numpy.ndarray): Array with one-hot encoded representation of target values.
    """
    n_classes = 6
    samples = np.zeros((x.shape[0], n_classes))

    for i in range(target.shape[0]):
        samples[i][int(target[i])] = 1

    return samples


def load_AtrialFibrillation(persistant=False):
    """
    Load arrhythmia dataset and perform preprocessing.

    Args:
        persistant (bool): If True, keeps the downloaded dataset files.
                           If False, deletes the dataset files after loading.
                           Default is False.

    Returns:
        data (dict): Dictionary containing the data for the MNIST dataset.
    """

    # Load the data
    url = "https://raw.githubusercontent.com/NahuelCostaCortez/RVAE/main/data/arrhythmia_data.npy"
    filename = "arrhythmia_data.npy"
    data_dir = os.path.join("..", "datasets", "AtrialFibrilation")

    # Create a directory to store the downloaded files
    os.makedirs(data_dir, exist_ok=True)

    target_path = os.path.join(data_dir, filename)

    if not os.path.exists(target_path):
        Logger().log_info(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, target_path)
    else:
        Logger().log_info(f"Skipping... Data already exists.")

    data = np.load(target_path, allow_pickle=True).item()

    # Split into a train, validation, test
    x_train = data["input_train"]
    x_val = data["input_vali"]
    x_test = data["input_test"]

    y_train = data["target_train"]
    y_val = data["target_vali"]
    y_test = data["target_test"]

    # If required delete data
    if not persistant:
        Logger().log_info("Deleting arrhythmia data...")
        rmtree(data_dir)

    data = {}
    data["x_train"] = x_train
    data["x_val"] = x_val
    data["x_test"] = x_test
    data["y_train"] = y_train
    data["y_val"] = y_val
    data["y_test"] = y_test

    return data


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
        "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/train_"
        + subset
        + ".txt"
    )
    url_RUL = (
        "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/RUL_"
        + subset
        + ".txt"
    )
    url_test = (
        "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/test_"
        + subset
        + ".txt"
    )

    filenames = [
        "train_" + subset + ".txt",
        "RUL_" + subset + ".txt",
        "test_" + subset + ".txt",
    ]
    data_dir = os.path.join("..", "datasets", "CMAPSS")

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


def load_dataset(dataset, persistant=False):
    """
    Load the dataset.

    Args:
        dataset (str): Name of the dataset to be loaded.
                    If False, returns the raw data.
        persistant (bool): If True, keeps the downloaded dataset files.
                           If False, deletes the dataset files after loading.
                           Default is False.
    """
    if dataset == "MNIST":
        return load_MNIST(persistant)
    elif dataset == "AtrialFibrillation":
        return load_AtrialFibrillation(persistant)
    elif dataset == "CMAPSS":
        return load_CMAPSS(persistant)
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
    # exclude the load_dataset function
    dataset_names.remove("dataset")
    # remove duplicates
    dataset_names = list(set(dataset_names))
    return dataset_names
