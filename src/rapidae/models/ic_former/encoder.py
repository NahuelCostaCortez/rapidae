from rapidae.models.base import BaseEncoder
from keras.layers import Layer, Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout, Conv1D
from keras import Sequential
import tensorflow as tf

# currently only available in tensorflow due to "positions" variable
class PatchEmbedding(Layer):
    def __init__(self, num_patches, projection_dim):
        """
        num_patches: number of patches == number of cycles
        projection_dim: dimension of the embedding
        """
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
                input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        """
        Applies positional embedding to know the cycle order
        patch: [batch_size, num_patches, seq_len]
        
        returns: [batch_size, num_patches, projection_dim]
        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class EncoderLayer(Layer):
    def __init__(self, num_heads, head_size, dff, latent_dim, rate=0.1):
        """
        num_heads: number of heads in the multi-head attention layer
        head_size: dimension of the head embedding
        dff: dimension of the feed forward network
        latent_dim: dimension of the feed foward output
        rate: dropout rate
        """
        super(EncoderLayer, self).__init__()

        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=rate)
        self.dropout1 = Dropout(rate)
        self.ffn = Sequential([
                                LayerNormalization(epsilon=1e-6),
                                Conv1D(filters=dff, kernel_size=1, activation="relu"),
                                Conv1D(filters=latent_dim, kernel_size=1)
                            ])

    def call(self, inputs, training, mask):
        """
        inputs: [batch_size, num_patches, d_model]
        training: boolean, specifies whether the dropout layer should behave in training mode (adding dropout) or in inference mode (doing nothing)
        mask: padding mask in the multi-head attention layer
        
        returns: [batch_size, num_patches, d_model]
        """
        x = self.layernorm(inputs)
        attn_output, att_weights = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=True, training=training)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        res = attn_output + inputs
        ffn_output = self.ffn(res)  # (batch_size, input_seq_len, d_model)

        return ffn_output + res, attn_output, att_weights, res

class Encoder(BaseEncoder):
    def __init__(self, input_dim=12, num_layers=2, num_heads=2, head_size=32, dff=32, latent_dim=128, rate=0.1):
        """
        input_dim: dimension of the input data
        num_layers: number of encoder layers
        num_heads: number of heads in the multi-head attention layer
        head_size: dimension of the head embedding
        dff: dimension of the feed forward network
        latent_dim: latent dimension of the encoder
        """
        BaseEncoder.__init__(self, input_dim, latent_dim)

        self.patch_embedding = PatchEmbedding(input_dim, latent_dim)

        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(num_heads, head_size, dff, latent_dim, rate) for _ in range(num_layers)]

    def call(self, x, training, mask):

        """
        x: [batch_size, num_patches, d_model]
        training: boolean, specifies whether the dropout layer should behave in training mode (adding dropout) or in inference mode (doing nothing)
        mask: padding mask in the multi-head attention layer
        
        return: [batch_size, num_patches, d_model]
        """
        x = self.patch_embedding(x)

        attention_weights = {}
        attention_outputs = {}
        attention_outputs_sum = {}

        for i in range(self.num_layers):
            x, att_output, att_weights, att_output_sum = self.enc_layers[i](x, training=training, mask=mask)
            attention_outputs[f'encoder_layer{i+1}'] = att_output
            attention_weights[f'encoder_layer{i+1}'] = att_weights
            attention_outputs_sum[f'encoder_layer{i+1}'] = att_output_sum

        return x, attention_outputs, attention_weights, attention_outputs_sum