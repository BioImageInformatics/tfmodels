import tensorflow as tf
import numpy as np
import sys, os

from base_discriminator import BaseDiscriminator
from ops import (
    lrelu,
    linear,
    conv,
    deconv, )

class Discriminator(BaseDiscriminator):
    ## Overload the base class.. do I even need the base class?
    ## TODO expose number of kernels and number of upsample steps to the world
    discriminator_defaults = {
        'sess': None,
        'dataset': None
        'strides': [4, 2, 2, 2],
        'z_dim': 64,
        'dis_kernels': [32, 64, 128, 256],
    }

    def __init__(self, **kwargs):
        self.discriminator_defaults.update(**kwargs)
        super(BaseDiscriminator, self).__init__(**self.discriminator_defaults)

        assert len(self.dis_kernels) == len(self.strides)

    def model(self, x_in, keep_prob=0.5, reuse=False):
        print 'Discriminator'
        print 'Nonlinearity: ', self.nonlin
        nonlin = self.nonlin

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            print '\t x_in:', x_in.get_shape()

            ## Conv net
            c0 = nonlin(conv(x_in, self.dis_kernels[0], k_size=5, stride=self.strides[0], var_scope='c0'))
            c0 = tf.contrib.nn.alpha_dropout(c0, keep_prob=keep_prob) ## 64 x 64
            print '\t c0', c0.get_shape()

            c1 = nonlin(conv(c0, self.dis_kernels[1], k_size=5, stride=self.strides[1], var_scope='c1'))
            c1 = tf.contrib.nn.alpha_dropout(c1, keep_prob=keep_prob) ## 32 x 32
            print '\t c1', c1.get_shape()

            c2 = nonlin(conv(c1, self.dis_kernels[2], k_size=5, stride=self.strides[2], var_scope='c2'))
            c2 = tf.contrib.nn.alpha_dropout(c2, keep_prob=keep_prob) ## 16 x 16
            print '\t c2', c2.get_shape()

            c3 = nonlin(conv(c2, self.dis_kernels[3], k_size=5, stride=self.strides[3], var_scope='c3'))
            c3 = tf.contrib.nn.alpha_dropout(c3, keep_prob=keep_prob) ## 8 x 8
            print '\t c3', c3.get_shape()

            flat = tf.contrib.layers.flatten(c3)
            print '\t flat', flat.get_shape()

            lin1 = nonlin(linear(flat, 512, var_scope='lin1'))
            lin2 = nonlin(linear(lin1, 64, var_scope='lin2'))
            p_real = linear(lin2, 1, var_scope='p_real')

            return p_real
