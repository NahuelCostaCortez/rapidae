from rapidae.models.base import BaseEncoder
from keras.layers import SimpleRNN, Dense


class Encoder(BaseEncoder):
    def __init__(self, input_dim, latent_dim, embed_dim=100):
        BaseEncoder.__init__(self, input_dim, latent_dim)
        self.embed_dim = embed_dim
        self.hidden_dim = 2 * embed_dim

        self.rnn1 = SimpleRNN(self.hidden_dim, return_state=True, return_sequences=True)
        self.rnn2 = SimpleRNN(self.embed_dim, return_state=True, return_sequences=True)

        self.fc_mu = Dense(self.latent_dim)
        self.fc_var = Dense(self.latent_dim)

    def call(self, x):
        # (batch_size, seq_len, hidden_dim), (batch_size, hidden_dim)
        x, _ = self.rnn1(x)
        # (batch_size, seq_len, embed_dim), (batch_size, embed_dim)
        x, hidden_state = self.rnn2(x)
        # (batch_size, latent_dim)
        mu = self.fc_mu(hidden_state)
        # (batch_size, latent_dim)
        log_var = self.fc_var(hidden_state)
        return mu, log_var
