import tensorflow as tf
import numpy as np
import sys, os

from base_generator import BaseGenerator
from ops import (
    lrelu,
    linear,
    conv,
    deconv, )

class Generator(BaseGenerator):
    ## Overload the base class.. do I even need the base class?
    ## TODO expose number of kernels and number of upsample steps to the world
    generator_defaults = {
        'sess': None,
        'dataset': None
        'n_upsamples': 4,
        'x_dims': [256, 256, 3],
        'gen_kernels': [32, 64, 128, 256],
    }

    def __init__(self, **kwargs):
        self.generator_defaults.update(**kwargs)
        super(BaseGenerator, self).__init__(**self.generator_defaults)

        assert len(self.gen_kernels) == self.n_upsamples

    def model(self, z_in, keep_prob=0.5, reuse=False):
        print 'Generator'
        print 'Nonlinearity: ', self.nonlin
        nonlin = self.nonlin

        with tf.variable_scope('Generator') as scope:
            if reuse:
                scope.reuse_variables()

            ## Project z
            print '\t z_in:', z_in.get_shape()
            target_edge = self.x_dims[0] / (2**self.n_upsamples)
            target_dim = target_edge * target_edge * self.gen_kernels[3]
            print '\t target_dim:', target_dim.get_shape()
            projection = nonlin(linear(z_in, target_dim, var_scope='projection'))
            projection = tf.reshape(projection, [-1, target_edge, target_edge, self.gen_kernels[3]])
            print '\t projection:', projection.get_shape() ## 16 x 16

            ## Deconvolutions
            c2_0 = nonlin(conv(projection, self.gen_kernels[2], stride=1, var_scope='c2_0'))
            c2_1 = nonlin(conv(c2_0, self.gen_kernels[2], stride=1, var_scope='c2_1'))
            d2 = nonlin(deconv(c2_1, self.gen_kernels[2], var_scope='d2'))
            d2 = tf.contrib.nn.alpha_dropout(d2, keep_prob=keep_prob)
            print '\t d2:', d2.get_shape() ## 32 x 32

            c1_0 = nonlin(conv(d2, self.gen_kernels[1], stride=1, var_scope='c1_0'))
            c1_1 = nonlin(conv(c1_0, self.gen_kernels[1], stride=1, var_scope='c1_1'))
            d1 = nonlin(deconv(c1_1, self.gen_kernels[1], var_scope='d1'))
            d1 = tf.contrib.nn.alpha_dropout(d1, keep_prob=keep_prob)
            print '\t d1:', d1.get_shape() ## 64 x 64

            c0_0 = nonlin(conv(d1, self.gen_kernels[0], stride=1, var_scope='c0_0'))
            c0_1 = nonlin(conv(c0_0, self.gen_kernels[0], stride=1, var_scope='c0_1'))
            d0 = nonlin(deconv(c0_1, self.gen_kernels[0], var_scope='d0'))
            d0 = tf.contrib.nn.alpha_dropout(d0, keep_prob=keep_prob)
            print '\t d0:', d0.get_shape() ## 128 x 128

            ## Output
            c_out_0 = nonlin(conv(d0, self.gen_kernels[0], stride=1, var_scope='c_out_0'))
            c_out_1 = nonlin(conv(c_out_0, self.gen_kernels[0], stride=1, var_scope='c_out_1'))
            x_hat = tf.nn.sigmoid(deconv(c_out_1, self.x_dims[2], var_scope='x_hat'))
            print '\t x_hat:', x_hat.get_shape() ## 256 x 256

            return x_hat
