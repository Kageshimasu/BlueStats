import tensorflow as tf 
import numpy as np

from common.base import BaseModel
from util.layers import *

class VAE(BaseModel):

    def __init__(self, session, size=(256, 256), code_size=2):
        super().__init__()
        self._inputs = tf.placeholder(tf.float32, (None, size[0], size[1]))
        self._session = session
        encoder = tf.make_template('encoder', self._encode)
        decoder = tf.make_template('encoder', self._decode)
        # encode
        posterior = encoder(self._inputs, code_size)
        self._code = posterior.sample()
        # calc loss
        prior = self._make_prior(code_size)
        divergence = tf.contrib.distributions.kl_divergence(posterior, prior)
        likelihood = decoder(self._code, [size[0], size[1]]).log_prob(self._inputs)
        self._loss_elbo = -tf.reduce_mean(likelihood - divergence)
        # decode random variable to get samples
        self._sample_images = decoder(prior.sample(10), [size[0], size[1]]).mean()
        self._optimizer = tf.train.AdamOptimizer(1e-3).minimize(self._loss_elbo)
        self._session.run(tf.global_variables_initializer())

    def train(self, inputs):
        self._session.run(
            self._optimizer,
            feed_dict={
                self._inputs: inputs,
                }
            )
    
    def predict(self, inputs):
        return self._session.run(
            self._sample_images,
            feed_dict={
                self._inputs: inputs,
                }
            )
    
    def get_code(self, inputs):
        return self._session.run(
            self._code,
            feed_dict={
                self._inputs: inputs,
            }
        )
    
    def evaluate(self, inputs):
        loss = self._session.run(
            self._loss_elbo,
            feed_dict={
                self._inputs: inputs,
                }
            )
        evaluated_list = {
        "elbo": loss,
        }
        return evaluated_list
    
    def save(self, file_path):
        saver = tf.train.Saver()
        saver.save(self._session, file_path)

    def load(self, file_path):
        saver = tf.train.Saver()
        saver.restore(self._session, file_path)

    def _encode(self, x, code_size):
        x = tf.layers.flatten(x)
        for i in range(1, 3):
            x = dense(x, 200)
        loc = tf.layers.dense(x, code_size)
        scale = tf.layers.dense(x, code_size, activation=tf.nn.softplus)
        return tf.contrib.distributions.MultivariateNormalDiag(loc, scale)

    def _decode(self, code, data_shape):
        for i in range(1, 3):
            code = dense(code, 200)
        code = dense(code, np.prod(data_shape))
        y = tf.reshape(code, [-1] + data_shape)
        return tf.contrib.distributions.Independent(tf.contrib.distributions.Bernoulli(y), 2)
    
    def _make_prior(self, code_size):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        return tf.contrib.distributions.MultivariateNormalDiag(loc, scale)
    
    def _predict(self, inputs):
        pass