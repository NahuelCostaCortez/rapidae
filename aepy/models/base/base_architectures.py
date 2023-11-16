"""
Base class for defining the encoder and decoder architectures
"""

import keras_core as keras
import tensorflow as tf
from keras_core import layers


class BaseEncoder(layers.Layer):
	"""
	Base class for all encoders
	"""

	def __init__(self, input_dim, latent_dim, layers_conf, name="encoder"):
		super().__init__(name=name)
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.layers_conf = layers_conf
		self.z_mean = layers.Dense(self.latent_dim, name="z_mean")
		self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")

	def call(self, x):
		""" 
		This function must be implemented in a child class.

		Parameters
		----------
			x (tf.Tensor): input tensor

		Returns
		-------
			tf.Tensor: the output of the encoder
		"""
		raise NotImplementedError


class BaseDecoder(layers.Layer):
	"""
 	Base class for all decoders
	"""

	def __init__(self, input_dim, latent_dim, layers_conf, name="decoder"):
		super().__init__(name=name)
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.layers_conf = reversed(layers_conf)

	def call(self, x):
		""" 
  		This function must be implemented in a child class 
		  
		Parameters
		----------
			x (tf.Tensor): input tensor with the latent data

		Returns
		-------
			tf.Tensor: the output of the decoder
		"""
		raise NotImplementedError
		
