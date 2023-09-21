"""
Class to load some common datasets.
"""
import numpy as np
import pandas as pd
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


def load_MNIST():
	"""
	Returns the train, y_train and test for the MNIST dataset.
	
	These are downloaded from the Keras repository:
	"""
 
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = np.expand_dims(x_train, axis=-1)
	x_test = np.expand_dims(x_test, axis=-1)
	return x_train, y_train, x_test, y_test


def load_CMAPSS(subset="FD001"):
	"""
	Returns train, test, y_test for the requested subset of the CMAPSS dataset.
 
	These are download from the NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational repository 
	since they are not longer available in the original source:
	https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq
	"""
 
	if subset not in ["FD001", "FD002", "FD003", "FD004"]:
		raise ValueError("Invalid subset. Supported subsets are: FD001, FD002, FD003, FD004")

	# Load the data
	url_train = "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/train_" + subset + ".txt"
	url_RUL = "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/RUL_" + subset + ".txt"
	url_test = "https://raw.githubusercontent.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational/main/data/test_" + subset + ".txt"

	# columns
	index_names = ['unit_nr', 'time_cycles']
	setting_names = ['setting_1', 'setting_2', 'setting_3']
	sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
	col_names = index_names + setting_names + sensor_names

	train = pd.read_csv(url_train, sep=r'\s+', header=None, 
					 names=col_names)
	test = pd.read_csv(url_test, sep=r'\s+', header=None, 
					 names=col_names)
	y_test = pd.read_csv(url_RUL, sep=r'\s+', header=None, 
					 names=['RemainingUsefulLife'])

	return train, test, y_test
  


    