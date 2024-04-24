from rapidae.models.base import BaseDecoder
from keras.layers import Dense, SimpleRNN, RepeatVector
from keras.ops import reshape


class Decoder(BaseDecoder):
    def __init__(self, input_dim, latent_dim, embed_dim=32):
        BaseDecoder.__init__(self, input_dim, latent_dim)
        self.seq_len = input_dim[0]
        self.embed_dim = embed_dim
        self.hidden_dim = 2 * embed_dim
        self.feat_dim = input_dim[1]

        self.fc = Dense(embed_dim)
        self.rnn1 = SimpleRNN(self.embed_dim, return_state=True, return_sequences=True)
        self.rnn2 = SimpleRNN(self.hidden_dim, return_state=True, return_sequences=True)
        self.output_layer_mean = Dense(self.feat_dim)
        self.output_layer_log_var = Dense(self.feat_dim)

    def call(self, x):
        # (batch_size, embed_dim)
        x = self.fc(x)
        # (batch_size, seq_len, embed_dim)
        x = RepeatVector(self.seq_len)(x)
        # (batch_size, seq_len, hidden_dim)
        x, _ = self.rnn1(x)
        # (batch_size, seq_len, embed_dim)
        x, _ = self.rnn2(x)
        # (batch_size * seq_len, embed_dim)
        x = reshape(x, (-1, self.hidden_dim))
        # (batch_size * seq_len, feat_dim)
        x_mean = self.output_layer_mean(x)
        # (batch_size, seq_len, feat_dim)
        x_mean = reshape(x_mean, (-1, self.seq_len, self.feat_dim))
        # (batch_size * seq_len, feat_dim)
        x_log_var = self.output_layer_log_var(x)
        # (batch_size, seq_len, feat_dim)
        x_log_var = reshape(x_log_var, (-1, self.seq_len, self.feat_dim))
        return x_mean, x_log_var
