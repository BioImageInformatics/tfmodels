import tensorflow as tf
import numpy as np
import sys, os

sys.path.insert(0, '../tfmodels')
from tfmodels import (VAE, BaseEncoder, BaseGenerator)
from tfmodels import (conv,
                      deconv,
                      linear,
                      conv_cond_concat)

"""
Implement the customized Generator.model()
"""
class Encoder(BaseEncoder):
    vae_encoder_defaults = {
        'enc_kernels': [32, 64, 128],
        'z_dim': 64,
    }

    def __init__(self, **kwargs):
        self.vae_encoder_defaults.update(**kwargs)
        super(Encoder, self).__init__(**self.vae_encoder_defaults)

    def model(self, x_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up mnistVAE/Discriminator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            print '\t x_in', x_in.get_shape()
            net = nonlin(conv(x_in, self.enc_kernels[0], k_size=5, stride=2, var_scope='e0'))
            net = nonlin(conv(net,  self.enc_kernels[1], k_size=5, stride=2, var_scope='e1'))
            net = nonlin(conv(net,  self.enc_kernels[2], k_size=3, stride=2, var_scope='e2'))

            flat = tf.contrib.layers.flatten(net)
            flat_dropout = tf.contrib.nn.alpha_dropout(flat, keep_prob=keep_prob)
            net = nonlin(linear(flat_dropout, 512, var_scope='e3'))

            mu = linear(net, self.z_dim, var_scope='mu')
            log_var = linear(net, self.z_dim, var_scope='log_var')

            return mu, log_var

"""
Implement the customized Generator.model()
"""
class Generator(BaseGenerator):
    vae_generator_defaults = {
        'gen_kernels': [128, 64, 32],
        'x_dims': [28, 28, 1],}

    def __init__(self, **kwargs):
        self.vae_generator_defaults.update(**kwargs)
        super(Generator, self).__init__(**self.vae_generator_defaults)

    def model(self, z_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up mnistVAE/Generator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            ## These first two layers will be pretty much the same in all generators
            ## Project
            print '\t z_in', z_in.get_shape()
            z_dim = z_in.get_shape().as_list()[-1]
            net = tf.reshape(z_in, (-1, 1, 1, z_dim))
            net = nonlin(deconv(net, self.gen_kernels[0], upsample_rate=4, k_size=4, var_scope='g0'))
            net = nonlin(deconv(net, self.gen_kernels[1], upsample_rate=2, k_size=4, var_scope='g1'))
            net = nonlin(deconv(net, self.gen_kernels[2], upsample_rate=2, k_size=3, var_scope='g2'))
            x_hat = deconv(net, self.x_dims[-1], upsample_rate=2, k_size=3, var_scope='x_hat')
            x_hat = tf.image.resize_image_with_crop_or_pad(x_hat, self.x_dims[0], self.x_dims[1])

            return x_hat


class mnistVAE(VAE):
    mnistVAE_defaults = {
        'enc_kernels': [64, 128, 512],
        'gen_kernels': [256, 128, 64],
        'x_dims': [28, 28, 1],
        'z_dim': 2
    }
    def __init__(self, **kwargs):
        self.mnistVAE_defaults.update(**kwargs)

        ## Initialize the encoder and generator classes
        encoder = Encoder(enc_kernels = self.mnistVAE_defaults['enc_kernels'],
            z_dim= self.mnistVAE_defaults['z_dim'])
        generator = Generator(gen_kernels= self.mnistVAE_defaults['gen_kernels'],
            x_dims= self.mnistVAE_defaults['x_dims'])

        ## Pass in the custom encoder and generator
        self.mnistVAE_defaults['encoder'] = encoder
        self.mnistVAE_defaults['generator'] = generator

        super(mnistVAE, self).__init__(**self.mnistVAE_defaults)


    """
    Here we can have custom inference and training functions by overloading the right methods
    """
    def inference(self, z_values):
        ## Take in values for z and return p(data|z)
        feed_dict = {self.zed: z_values, self.keep_prob: 1.0}
        x_hat = self.sess.run(self.x_hat, feed_dict=feed_dict)
        return x_hat


    def train_step(self):
        self.global_step += 1
        # if self.iterator_dataset:
        feed_dict = {self.x_in: next(self.dataset.iterator)}
        self.sess.run(self.train_op, feed_dict=feed_dict)
        # else:
        #     self.sess.run(self.train_op)

        if self.global_step % self.summary_iters == 0:
            summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_str, self.global_step)
