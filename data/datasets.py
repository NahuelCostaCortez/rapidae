"""
Class to load some common datasets.
"""
import os
from os.path import isfile
import urllib
import numpy as np
import pandas as pd
import gzip
from tensorflow.keras.datasets import mnist
import requests


def get_data_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        return data
    else:
        print("Failed to retrieve data from", url)
        return None


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        #data = np.expand_dims(data, axis=-1)
        #data = np.expand_dims(data, axis=-1)
    return data.reshape(-1, 28, 28, 1)


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)
    return data


def load_MNIST(use_keras=False):
    """
    Returns the train, y_train and test data for the MNIST dataset.
    It can be obtained from original source or from Keras repository.
    """

    if use_keras:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    else:
        # url
        url_base = 'http://yann.lecun.com/exdb/mnist/'
        filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        data_dir = 'mnist_data'

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
                print(f'Downloading {filename}...')
                urllib.request.urlretrieve(url, target_path)
            else:
                print(f'{filename} already exists.')

        # Load the training and test data
        x_train = load_mnist_images(train_img_path)
        y_train = load_mnist_labels(train_lbl_path)
        x_test = load_mnist_images(test_img_path)
        y_test = load_mnist_labels(test_lbl_path)

    return x_train, y_train, x_test, y_test


def load_CMAPSS(subset="FD001"):
    """
    Returns train, test, y_test for the requested subset of the CMAPSS dataset.

    These are download from the NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational repository 
    since they are not longer available in the original source:
    https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq
    """

    if subset not in ["FD001", "FD002", "FD003", "FD004"]:
        raise ValueError(
            "Invalid subset. Supported subsets are: FD001, FD002, FD003, FD004")

    # Load the data
    url_train = "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/train_" + subset + ".txt"
    url_RUL = "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/RUL_" + subset + ".txt"
    url_test = "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/test_" + subset + ".txt"

    # columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv(url_train, sep=r'\s+', header=None,
                        names=col_names)
    test = pd.read_csv(url_test, sep=r'\s+', header=None,
                                     names=col_names)
    y_test = pd.read_csv(url_RUL, sep=r'\s+', header=None,
                         names=['RemainingUsefulLife'])

    return train, test, y_test
