import keras
import math


class Normal:
    def __init__(self, mu, std, seed=None, **kwargs):
        """
        Initialize a Distribution object.

        Args:
            mu (float): The mean of the distribution.
            std (float): The standard deviation of the distribution.
            seed (int, optional): The seed value for random number generation. An integer or instance of keras.random.SeedGenerator. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base class.

        Notes:
            - This class expects the standard deviation (`std`) as input.
            - If you have the variance, you need to convert it to standard deviation first:
                `std = ops.sqrt(var)`
            - If you have the log probability, you need to convert it to standard deviation first:
                `std = ops.exp(0.5 * log_var)`
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.std = std
        self.seed = seed
        # self.seed_generator = keras.random.SeedGenerator(seed)

    def log_prob(self, x):
        """
        Calculates the logarithm of the probability density function (PDF) at the given value.

        Parameters:
            x (float): The value at which to calculate the logarithm of the PDF.

        Returns:
            float: The logarithm of the PDF at the given value.
        """
        var = self.std**2
        log_scale = keras.ops.log(self.std)
        logp = (
            -((x - self.mu) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

        return logp

    def sample(self):
        """
        Uses (mu, std) to sample.

        Returns:
            keras.tensor: Samples generated from the distribution.
        """
        batch = keras.ops.shape(self.mu)[0]
        dim = keras.ops.shape(self.mu)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed)

        return self.mu + self.std * epsilon


'''
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a sample."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = random.SeedGenerator(1337)

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = ops.shape(z_mean)[0]
            dim = ops.shape(z_mean)[1]
            # Added seed for reproducibility
            epsilon = random.normal(shape=(batch, dim), seed=self.seed_generator)

            return z_mean + ops.exp(0.5 * z_log_var) * epsilon
'''


class Logistic:
    def __init__(self, mu, scale, seed=1337, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.std = scale
        self.seed_generator = keras.random.SeedGenerator(seed)

    def log_prob(self, x):
        """
        Calculates the logarithm of the probability density function (PDF) at the given value.

        Parameters:
            x (float): The value at which to calculate the logarithm of the PDF.

        Returns:
            float: The logarithm of the PDF at the given value.
        """
        y = -(x - self.mu) / self.scale
        logp = -y - keras.ops.log(self.scale) - 2 * keras.activations.softplus(-y)
        return logp

    def logistic_eps(self, shape, bound=1e-5):
        # first sample from a Gaussian
        u = keras.random.normal(shape=shape, seed=self.seed_generator)

        # clip to interval [bound, 1-bound] to ensure numerical stability
        # (to avoid saturation regions of the sigmoid)
        u = keras.ops.clip(u, x_min=bound, x_max=1 - bound)

        # transform to a sample from the Logistic distribution
        # log(u / (1 - u))
        epsilon = keras.ops.log(u) - keras.ops.log1p(-u)

        return epsilon

    def sample(self):
        """
        Sample from the logistic distribution.
        Uses inverse sampling: instead of picking a random variable x that is
        most probable to be picked under L(mu, scale) - as we do for normal distribution -
        N(mu, sigma^2) -> x = mu + ops.exp(0.5 * log_var),
        we pick a random variable x that is most probable under the logistic PDF, that is,
        the inverse CDF (quantile funtion):

        x = mu + scale * log(u / (1 - u))

        This is done because the CDF of the logistic distribution is a sigmoid,
        and this makes training and sampling easier.

        Returns:
            keras.tensor: Samples generated from the distribution.
        """
        epsilon = self.logistic_eps(shape=keras.ops.shape(self.mu))
        return self.mu + self.scale * epsilon
