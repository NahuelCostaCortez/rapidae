from rapidae.models.base import BaseEncoder
from keras.layers import Masking, LSTM, Bidirectional, Dense


class Encoder(BaseEncoder):
    def __init__(self, input_dim, latent_dim, nz=1, embed_dim=100, **kwargs):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.nz = nz
        self.embed_dim = embed_dim
        self.hidden_dim = 2 * embed_dim

        # BOTTOM LATENT LAYER
        self.masking_value = (
            kwargs["masking_value"] if "masking_value" in kwargs else -99.0
        )
        self.mask = Masking(mask_value=self.masking_value)

        self.rnn1 = Bidirectional(
            LSTM(self.hidden_dim, return_sequences=True)
        )  # this returns output, hidden_state_f, cell_state_f, hidden_state_b, cell_state_b
        self.rnn2 = Bidirectional(LSTM(self.embed_dim, return_state=True))

        self.fc_mu = Dense(self.latent_dim)
        self.fc_var = Dense(self.latent_dim)

    def call(self, x, **kwargs):
        lvl = kwargs["lvl"] if "lvl" in kwargs else 0

        # bottom latent layer
        if lvl == 0:
            x = self.mask(x)
            # (batch_size, seq_len, hidden_dim*2)
            x = self.rnn1(x)
            # (batch_size, embed_dim), (batch_size, embed_dim), (batch_size, embed_dim), (batch_size, embed_dim), (batch_size, embed_dim)
            # (batch_size, seq_len, embed_dim), (batch_size, embed_dim), (batch_size, embed_dim), (batch_size, embed_dim), (batch_size, embed_dim) if return_sequences=True
            _, hidden_state_f, _, _, _ = self.rnn2(x)
            # (batch_size, latent_dim)
            mu = self.fc_mu(hidden_state_f)
            # (batch_size, latent_dim)
            log_var = self.fc_var(hidden_state_f)
        # deeper latent layers
        else:
            lvl -= 1

        return mu, log_var
