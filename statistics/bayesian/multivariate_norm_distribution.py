import numpy as np 

from ..common.base_bayesian_model import BayesianModel


class MultivariateNormalDistribution(BayesianModel):

    def __init__(self, dim):
        """
        train random variables ~ normal distribution
        the prediction distribution is T-Distribution
        :param dim:
        """
        if dim < 1:
            raise ValueError('dim must be higher than 1. but the dim is {}'.format(dim))
        self._mu = np.zeros(dim)
        self._beta = 0
        self._nu = 0
        self._w = np.zeros((dim, dim))
        self._dim = dim

    def update(self, x):
        """
        update the params
        :param x: (dim,) vector
        :return:
        """
        if x.shape[0] != self._dim:
            raise ValueError('inputs must be {} dim. but your inputs dim is {}'.format(self._dim, x.ndim))
        # calc new params
        n = 1
        beta = n + self._beta
        mu = 1 / beta * (x + self._beta * self._mu)
        nu = n + self._nu
        w = np.outer(x, x) + self._beta * np.outer(self._mu, self._mu) - beta * np.outer(mu, mu) + self._w
        # update the params
        self._mu = mu
        self._beta = beta
        self._nu = nu
        self._w = w

    def get_sample(self, n=1):
        """
        :param df:
        :param n:
        :return:
        """
        m = np.asarray(self._mu)
        s = (1 + self._beta) / ((1 - self._dim + self._nu) * self._beta) * self._w
        d = len(m)
        df = 1 + self._nu
        if df == np.inf:
            x = 1.
        else:
            x = np.random.chisquare(df, n) / df
        z = np.random.multivariate_normal(np.zeros(d), s, (n,))
        return (m + z / np.sqrt(x)).reshape(self._dim)

    def get_parameters(self):
        """
        :return:
        """
        return self._mu, (1 + self._beta) / ((1 - self._dim + self._nu) * self._beta) * self._w

    def pdf(self, x):
        """
        :param x:
        :return:
        """
        sigma = (1 + self._beta) / ((1 - self._dim + self._nu) * self._beta) * self._w
        inv_mu = np.dot((x - self._mu).T, np.linalg.inv(sigma))
        return (1 / np.sqrt(2 * np.pi) ** 2) / np.linalg.det(sigma) * np.exp(-1 / 2 * np.dot(inv_mu, (x - self._mu)))
