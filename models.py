import numpy as np
import theano
import theano.tensor
from collections import OrderedDict


def log_gaussian_symbolic(X, alpha, beta):
    alpha = alpha.dimshuffle('x', 0)
    X = X.dimshuffle(0, 'x')
    sqrt2pi = np.sqrt(2. * np.pi).astype('float32')
    lg = loggaussian_symbolic(X, alpha, beta)
    return (alpha * X + beta * X**2 +
            .5 * T.log(-2. * beta) +
            .5 * alpha**2 - np.log(sqrt2pi))

def gaussian_symbolic(X, alpha, beta):
    lg = log_gaussian_symbolic(X, alpha, beta)
    return T.exp(lg)

def mog_posterior_symbolic(X, alpha, beta, pi):
    gs = gaussian_symbolic(X, alpha, beta)
    numerator = pi.dimshuffle('x', 0) * gs
    return numerator / numerator.sum(axis=1, keepdims=True)

def mog_em_objective_symbolic(X, alpha, beta, pi):
    posterior = mog_posterior_symbolic(X, alpha, beta, pi)
    lg = log_gaussian_symbolic(X, alpha, beta)
    return (posterior * lg).sum()


class MixtureModel(object):
    """
    Mixture model base class.

    Parameters
    ----------
    n_mixtures : int
        Number of mixtures.
    seed : int (optional)
        Random seed.
    """
    def __init__(self, n_mixtures, seed=20161119):
        self.rng = np.random.RandomState(seed)
        self.n_mixtures = n_mixtures
        self._setup()

    def _setup(self):
        raise NotImplementedError

    def _update_X(self, X):
        self._update_X_theano(X.astype('float32'))

    def fit(self, X, n_steps=100):
        self._update_X(X)
        for ii in range(n_steps):
            print(self._update_params)

    def posterior(self, X):
        return self._posterior(X.astype('float32'))

    def em_objective(self, X):
        return self._em_objective(X.astype('float32'))


class GaussianMixture(MixtureModel):
    def _setup(self):
        pi = self.rng.rand(n_mixtures)
        pi /= pi.sum()
        self.pi = theano.shared(pi.astype('float32'))
        alpha = self.rng.randn(n_mixtures)
        self.alpha = theano.shared(alpha.astype('float32'))
        beta = np.ones(n_mixtures)
        self.beta = theano.shared(beta.astype('float32'))

        self.X = theano.shared(np.ones((1, 1)).astype('float32'))
        X = T.matrix('X')
        updates = OrderedDict()
        updates[self.X] = X
        self._update_X_theano = theano.function([X], [], updates=updates)

        # Setup posterior symbolic and theano function with input X
        posterior = self._posterior_symbolic(X, self.alpha,
                                             self.beta, self.pi)
        self._posterior = theano.function([X], posterior)

        # Setup EM objective symbolic and theano function
        em_objective = self._em_objective_symbolic(X, posterior, self.alpha,
                                                   self.neg_log_beta)
        self._em_objective = theano.function([X], em_objective)

        # Setup posterior symbolic with shared X for fitting
        posterior = mog_posterior_symbolic(self.X, self.alpha,
                                       self.beta, self.pi)
        em_objective = mog_em_objective_symbolic(self.X, posterior, self.alpha,
                                                 self.beta, self.pi)
        # Setup EM fit function
        updates = OrderedDict()
        pi_update = posterior.mean(axis=0)
        pi_update = T.switch(pi_update > 0., pi_update, 0.)
        pi_update = pi_update / pi_update.sum()
        alpha_update = posterior.T.dot(X) / posterior.sum(axis=0)
        beta_update = - .5 * posterior.sum(axis=0) / posterior.T.dot(X**2)
        updates[self.pi] = pi_update
        updates[self.alpha] = alpha_update
        updates[self.beta] = beta_update
        self._update_params = theano.function([], [em_objective], updates=updates)


class RayleighMixture(MixtureModel):
    def _setup(self):
        pi = self.rng.rand(n_mixtures)
        pi /= pi.sum()
        self.pi = theano.shared(pi.astype('float32'))
        neg_log_beta = self.rng.randn(n_mixtures)
        self.neg_log_beta = theano.shared(neg_log_beta.astype('float32'))
