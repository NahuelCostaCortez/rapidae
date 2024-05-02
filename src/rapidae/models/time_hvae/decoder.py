from rapidae.models.base import BaseDecoder
from keras.layers import Dense, RepeatVector, LSTM, Bidirectional
from keras.ops import reshape
from torch import nn, Tensor
import torch.nn.functional as F


class Decoder(BaseDecoder):
    def __init__(self, input_dim, latent_dim, embed_dim=32):
        BaseDecoder.__init__(self, input_dim, latent_dim)
        self.seq_len = input_dim[0]
        self.embed_dim = embed_dim
        self.hidden_dim = 2 * embed_dim
        self.feat_dim = input_dim[1]

        # BOTTOM LATENT LAYER
        self.fc = Dense(embed_dim)
        #self.repeat = RepeatVector(self.seq_len)
        self.rnn1 = Bidirectional(LSTM(self.embed_dim, return_sequences=True))
        self.rnn2 = LSTM(self.hidden_dim, return_sequences=True)
        self.x_mean = Dense(self.feat_dim)
        self.x_log_var = Dense(self.feat_dim)
        #self.x_mu = Dense(self.num_features, name="x_mu") -> logistica
        #self.x_std = nn.Parameter(Tensor(*(self.seq_len, self.num_features))) -> logistica
        #nn.init.zeros_(self.x_std) -> logistica

    def call(self, x, **kwargs):
        lvl = kwargs["lvl"] if "lvl" in kwargs else 0

        # bottom latent layer
        if lvl == 0:
            # (batch_size, embed_dim)
            x = self.fc(x)
            # (batch_size, seq_len, embed_dim)
            #x = self.repeat(x)
            # (batch_size, seq_len, embed_dim)
            x = self.rnn1(x)
            # (batch_size, seq_len, embed_dim)
            x = self.rnn2(x)
            # (batch_size * seq_len, embed_dim)
            #x = reshape(x, (-1, self.hidden_dim))
            # (batch_size * seq_len, feat_dim)
            x_mean = self.x_mean(x)
            # (batch_size, seq_len, feat_dim)
            #x_mean = reshape(x_mean, (-1, self.seq_len, self.feat_dim))
            # (batch_size * seq_len, feat_dim)
            x_log_var = self.x_log_var(x)
            # (batch_size, seq_len, feat_dim)
            #x_log_var = reshape(x_log_var, (-1, self.seq_len, self.feat_dim))

            # nuevo -> logistica
            #x_mu = self.x_mu(x)
            # scale parameter of the conditional Logistic distribution
            # set a minimal value for the scale parameter of the bottom generative model
            #x_scale = -F.logsigmoid(-self.x_std)
        # deeper latent layers
        else:
            lvl -= 1

        #return x_mu, x_scale -> logistica
        return x_mean, x_log_var
