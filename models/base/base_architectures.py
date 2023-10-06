"""
Base class for defining the encoder and decoder architectures
"""

import keras
import tensorflow as tf
from keras import layers


class BaseEncoder(layers.Layer):
	"""
	Base class for all encoders
	"""

	def __init__(self, args, name="encoder"):
		super().__init__(name=name)
		self.input_dim = args.input_dim
		self.latent_dim = args.latent_dim
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

	def __init__(self, args, name="decoder"):
		super().__init__(name=name)
		self.input_dim = args.input_dim
		self.latent_dim = args.latent_dim

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
		
