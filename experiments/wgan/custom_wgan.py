import tensorflow as tf
import numpy as np
import os

import sys
sys.path.insert(0, '../..')

from tfmodels import BaseGenerator
from tfmodels import BaseModel
from tfmodels import BaseDiscriminator
from tfmodels import WGAN
from tfmodels import (conv,
                      deconv,
                      linear,
                      conv_cond_concat)

"""
Wasserstein Generative Adversarial Networks

https://arxiv.org/pdf/1701.07875.pdf
https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
"""
class Critic(BaseDiscriminator):
    wgan_critic_defaults = {
        'dis_kernels': [64, 128, 256, 512],
        'name': 'wgan_critic'
    }

    def __init__(self, **kwargs):
        self.wgan_critic_defaults.update(**kwargs)
        super(Critic, self).__init__(**self.wgan_critic_defaults)


    def model(self, x_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up GAN/Discriminator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            print '\t x_in', x_in.get_shape()
            c0 = nonlin(conv(x_in, self.dis_kernels[0], k_size=5, stride=3, var_scope='c0', selu=1))
            c1 = nonlin(conv(c0, self.dis_kernels[1], k_size=5, stride=3, var_scope='c1', selu=1))
            c2 = nonlin(conv(c1, self.dis_kernels[2], k_size=5, stride=3, var_scope='c2', selu=1))
            flat = tf.contrib.layers.flatten(c2)

            print '\t flat', flat.get_shape()
            h0 = nonlin(linear(flat, self.dis_kernels[3], var_scope='h0', selu=1))
            p_real = linear(h0, 1, var_scope='p_real', no_bias=True)
            print '\t p_real', p_real.get_shape()

            return p_real



class Generator(BaseGenerator):
    wgan_generator_defaults = {
        'gen_kernels': [256, 128, 64],
        'x_dims': [128, 128, 3],
        'z_dim': None
    }

    def __init__(self, **kwargs):
        self.wgan_generator_defaults.update(**kwargs)

        ## Set the shape for the layer immediately after noise
        ## Manually specifying this sets up freedom for the rest of the layers
        self.project_shape = 3*3*256
        self.resize_shape = [-1, 1, 1, self.z_dim]
        super(Generator, self).__init__(**self.wgan_generator_defaults)

    def model(self, z_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up GAN/Generator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            ## Project
            print '\t z_in', z_in.get_shape()
            # projection = nonlin(linear(z_in, self.project_shape, var_scope='projection', selu=1))
            reshape_layer = tf.reshape(z_in, self.resize_shape)

            print '\t project_conv', project_conv.get_shape()
            h0_0 = nonlin(deconv(reshape_layer, self.z_dim, upsample_rate=3, var_scope='h0_0', selu=1))
            h0 = nonlin(deconv(h0_0, self.gen_kernels[0], upsample_rate=3, var_scope='h0', selu=1))
            h0_1 = nonlin(conv(h0, self.gen_kernels[0], k_size=3, stride=1, pad='VALID', var_scope='h0_1', selu=1))
            h0_1 = tf.contrib.nn.alpha_dropout(h0_1, 0.5)

            h1 = nonlin(deconv(h0_1, self.gen_kernels[1], var_scope='h1', selu=1))
            h1_1 = nonlin(conv(h1, self.gen_kernels[1], k_size=3, stride=1, pad='SAME', var_scope='h1_1', selu=1))
            h1_1 = tf.contrib.nn.alpha_dropout(h1_1, 0.5)

            h2 = nonlin(deconv(h1_1, self.gen_kernels[2], var_scope='h2', selu=1))
            h2_1 = nonlin(conv(h2, self.gen_kernels[2], k_size=3, stride=1, pad='SAME', var_scope='h2_1', selu=1))

            x_hat = conv(h2_1, self.x_dims[-1], k_size=2, stride=1, pad='SAME', var_scope='x_hat')
            print '\t x_hat', x_hat.get_shape()

            return x_hat
            # return x_hat



class DCWGAN(WGAN):
    dcwgan_defaults = {
        'gen_kernels': [256, 128, 64],
        'dis_kernels': [64, 128, 256, 512],
        'x_dims': [128, 128, 3]
    }

    def __init__(self, **kwargs):
        self.dcwgan_defaults.update(**kwargs)
        for key, attr in self.dcwgan_defaults.items():
            setattr(self, key, attr)

        self.generator = Generator(gen_kernels=self.gen_kernels,
            x_dims=self.x_dims, z_dim=self.z_dim)
        self.critic = Critic(dis_kernels=self.dis_kernels)

        ## IFFY
        self.dcwgan_defaults['critic'] = self.critic
        self.dcwgan_defaults['generator'] = self.generator

        super(DCWGAN, self).__init__(**self.dcwgan_defaults)
