"""
This code is a Keras implementation (only for tensorflow backend) of MINE: Mutual Information Neural Estimation (https://arxiv.org/pdf/1801.04062.pdf)
Author: Chengzhang Zhu
Email: kevin.zhu.china@gmail.com
Date: 2019-08-16
"""

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda
import keras.backend as K
import tensorflow as tf
import numpy as np


def mine_loss(args):
    t_xy = args[0]
    t_xy_bar = args[1]
    loss = -(K.mean(t_xy) - K.logsumexp(t_xy_bar) + K.log(tf.cast(K.shape(t_xy)[0], tf.float32)))
    return loss


def shuffle(y):
    return tf.random_shuffle(y)


class MINE(object):
    def __init__(self, x_dim=None, y_dim=None, network=None):
        self.model = None
        if network is None:
            assert x_dim is not None and y_dim is not None, 'x_dim and y_dim should be both given.'
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.network = self._build_network()
        else:
            assert isinstance(network, Model), 'the network should be defined as a Keras Model class'
            self.network = network

    def fit(self, x, y, epochs=50, batch_size=100, verbose=1):
        if self.model is None:
            self._build_mine()
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        else:
            assert len(y) == 1, 'only support that y is one target'
        inputs = x + y
        history = self.model.fit(x=inputs, epochs=epochs, batch_size=batch_size, verbose=verbose)
        fit_loss = history.history['loss']
        mutual_information = self.predict(x, y)
        return fit_loss, mutual_information

    def predict(self, x, y):
        assert self.model is not None, 'should fit model firstly'
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        else:
            assert len(y) == 1, 'only support that y is one target'
        inputs = x + y
        return np.mean(self.model.predict(x=inputs))

    def _build_mine(self):
        # construct MINE model
        x_input = self.network.inputs[0:-1]  # enable a complex x input
        y_input = self.network.inputs[-1]  # the last position in the input list should be y
        y_bar_input = Lambda(shuffle)(y_input)  # shuffle y input as y_bar
        t_xy = self.network(x_input + [y_input])
        t_xy_bar = self.network(x_input + [y_bar_input])
        loss = Lambda(mine_loss, name='mine_loss')([t_xy, t_xy_bar])
        output = Lambda(lambda x: -x)(loss)
        self.model = Model(inputs=x_input + [y_input], outputs=output, name='MINE_model')
        self.model.add_loss(loss)
        self.model.compile(optimizer='adam')

    def _build_network(self):
        # build a three-layer fully connected network with 100 units at each layer with ELU activation functions
        x = Input(shape=(self.x_dim,), name='network/x_input')
        y = Input(shape=(self.y_dim,), name='network/y_input')
        hidden = Concatenate(name='network/concatenate_layer')([x, y])
        for i in range(3):
            hidden = Dense(100, activation='elu', name='network/hidden_layer_{}'.format(i+1))(hidden)
        output = Dense(1)(hidden)
        model = Model(inputs=[x, y], outputs=output, name='statistics_network')
        return model
