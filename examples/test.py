import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import keras_core as keras
import tensorflow as tf

from aepy.models.base.default_architectures import Encoder_Conv_VQ_MNIST, Decoder_Conv_VQ_MNIST


class VectorQuantizer(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized, self.losses

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


if __name__ == "__main__":
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    """
    inputs = tf.random.normal(shape=(128, 7, 7, 32))
    print(inputs.shape)
    print(inputs)
    vq = VectorQuantizer(num_embeddings=64, embedding_dim=32)
    outputs = vq(inputs)
    print(outputs[0].shape)
    """

    inputs = tf.random.normal(shape=(128, 28, 28, 1))
    print(inputs.shape)
    print(inputs.shape[1])
    print(inputs.shape[2])
    enc = Encoder_Conv_VQ_MNIST(input_dim=(inputs.shape[1], inputs.shape[2]), latent_dim=2, layers_conf=[32, 64])
    encoded = enc(inputs)
    print(encoded.shape)

    vq = VectorQuantizer(num_embeddings=64, embedding_dim=enc.latent_dim)
    quantized_latents, vq_loss = vq(encoded)
    print(quantized_latents.shape)

    dec = Decoder_Conv_VQ_MNIST(input_dim=(inputs.shape[1], inputs.shape[2]), latent_dim=2, layers_conf=[32, 64])
    decoded = dec(quantized_latents)
    print(decoded.shape)

