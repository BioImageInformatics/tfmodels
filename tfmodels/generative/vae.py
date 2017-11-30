import tensorflow as tf
import numpy as np
import sys, os

from ..utilities.basemodel import BaseModel
from encoder_basemodel import BaseEncoder
from generator_basemodel import BaseGenerator
from ..utilities.ops import (
    conv,
    deconv,
    linear,
    conv_cond_concat
)

''' Variational Autoencoder

Kingma and Welling, 2013

'''
# class Discriminator(BaseDiscriminator):
#     vae_discriminator_defaults = {
#         'dis_kernels': [32, 64, 128]
#     }
#
#     def __init__(self, **kwargs):
#         self.vae_discriminator_defaults.update(**kwargs)
#         super(Discriminator, self).__init__(**self.vae_discriminator_defaults)
#
#
#     def model(self, x_in, keep_prob=0.5, reuse=False):
#         with tf.variable_scope(self.name) as scope:
#             if reuse:
#                 scope.reuse_variables()
#
#             print 'Setting up VAE/Discriminator'
#             print 'Nonlinearity: ', self.nonlin
#             nonlin = self.nonlin
#
#             print '\t x_in', x_in.get_shape()
#             c0 = nonlin(conv(x_in, self.dis_kernels[0], k_size=5, stride=3, var_scope='c0'))
#             print '\t c0', c0.get_shape()
#             c1 = nonlin(conv(c0, self.dis_kernels[1], k_size=5, stride=3, var_scope='c1'))
#             print '\t c1', c1.get_shape()
#             # c2 = nonlin(conv(c1, self.dis_kernels[1], k_size=5, stride=3, var_scope='c2'))
#             flat = tf.contrib.layers.flatten(c1)
#             print '\t flat', flat.get_shape()
#             h0 = nonlin(linear(flat, self.dis_kernels[2], var_scope='h0'))
#             print '\t h0', h0.get_shape()
#             p_real = linear(h0, 1, var_scope='p_real')
#             print '\t p_real', p_real.get_shape()
#
#             return p_real


# ## FOR Adversarial autoencoder
# class VectorDiscriminator(BaseDiscriminator):
#     vae_discriminator_defaults = {
#         'dis_vec_kernels': [128, 128, 64]
#     }
#
#     def __init__(self, **kwargs):
#         self.vae_discriminator_defaults.update(**kwargs)
#         super(VectorDiscriminator, self).__init__(**self.vae_discriminator_defaults)
#
#
#     def model(self, vec_in, keep_prob=0.5, reuse=False):
#         with tf.variable_scope(self.name) as scope:
#             if reuse:
#                 scope.reuse_variables()
#
#             print 'Setting up VAE/VectorDiscriminator'
#             print 'Nonlinearity: ', self.nonlin
#             nonlin = self.nonlin
#
#             print '\t vec_in', vec_in.get_shape()
#             c0 = nonlin(linear(vec_in, self.dis_vec_kernels[0], var_scope='c0'))
#             print '\t c0', c0.get_shape()
#             c1 = nonlin(linear(c0, self.dis_vec_kernels[1], var_scope='c1'))
#             print '\t c1', c1.get_shape()
#             h0 = nonlin(linear(c1, self.dis_vec_kernels[2], var_scope='h0'))
#             print '\t h0', h0.get_shape()
#             p_real = linear(h0, 1, var_scope='p_real')
#             print '\t p_real', p_real.get_shape()
#
#             return p_real


class Encoder(BaseEncoder):
    vae_discriminator_defaults = {
        'enc_kernels': [32, 64, 128],
        'z_dim': 64,
    }

    def __init__(self, **kwargs):
        self.vae_discriminator_defaults.update(**kwargs)
        super(Encoder, self).__init__(**self.vae_discriminator_defaults)


    def model(self, x_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up VAE/Discriminator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            batch_size = tf.shape(x_in)[0]
            print '\t x_in', x_in.get_shape()
            c0 = nonlin(conv(x_in, self.enc_kernels[0], k_size=5, stride=3, var_scope='c0'))
            print '\t c0', c0.get_shape()
            c1 = nonlin(conv(c0, self.enc_kernels[1], k_size=5, stride=3, var_scope='c1'))
            print '\t c1', c1.get_shape()
            # c2 = nonlin(conv(c1, self.dis_kernels[1], k_size=5, stride=3, var_scope='c2'))
            flat = tf.contrib.layers.flatten(c1)
            print '\t flat', flat.get_shape()
            h0 = nonlin(linear(flat, self.enc_kernels[2], var_scope='h0'))
            print '\t h0', h0.get_shape()

            mu = linear(h0, self.z_dim, var_scope='mu')
            print '\t mu', mu.get_shape()
            log_var = linear(h0, self.z_dim, var_scope='log_var')
            print '\t var', log_var.get_shape()

            epsilon = tf.random_normal(shape=[batch_size, self.z_dim], mean=0.0, stddev=1.0)
            zed = mu + epsilon * tf.sqrt(tf.exp(log_var))
            print '\t zed', zed.get_shape()

            return zed, mu, log_var


class Generator(BaseGenerator):
    vae_generator_defaults = {
        'gen_kernels': [128, 64, 32],
        'n_upsamples': 3,
        'x_dims': [128, 128, 3],
        # 'z_in': None
    }

    def __init__(self, **kwargs):
        self.vae_generator_defaults.update(**kwargs)
        super(Generator, self).__init__(**self.vae_generator_defaults)


    def model(self, z_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up VAE/Generator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            ## Project
            print '\t z_in', z_in.get_shape()
            projection = nonlin(linear(z_in, self.project_shape, var_scope='projection')) ## [16*16]
            print '\t projection', projection.get_shape()
            project_conv = tf.reshape(projection, self.resize_shape) ## [16, 16, 1]
            print '\t project_conv', project_conv.get_shape()
            h0 = nonlin(deconv(project_conv, self.gen_kernels[0], var_scope='h0')) ## [32, 32, 128]
            print '\t h0', h0.get_shape()
            h1 = nonlin(deconv(h0, self.gen_kernels[1], var_scope='h1')) ## [64, 64, 64]
            print '\t h1', h1.get_shape()

            x_hat = tf.nn.sigmoid(conv(h1, self.x_dims[-1], stride=1, var_scope='x_hat'))
            print '\t x_hat', x_hat.get_shape()

            return x_hat





class VAE(BaseModel):
    vae_defaults = {
        'batch_size': 128,
        'dataset': None,
        'enc_kernels': [32, 64, 128],
        'gen_kernels': [32, 64, 128, 256],
        'global_step': 0,
        'iterator_dataset': False,
        'learning_rate': 1e-4,
        'log_dir': None,
        'mode': 'TRAIN',
        'name': 'vae',
        'save_dir': None,
        'sess': None,
        'summary_iters': 50,
        'x_dims': [256, 256, 3],
        'z_dim': 16, }

    def __init__(self, **kwargs):
        self.vae_defaults.update(**kwargs)
        super(VAE, self).__init__(**self.vae_defaults)

        assert self.sess is not None
        assert len(self.x_dims) == 3
        if self.mode=='TRAIN': assert self.dataset is not None

        self.encoder = Encoder(
            enc_kernels=self.enc_kernels,
            z_dims=self.z_dims )
        self.generator = Generator(
            gen_kernels=self.gen_kernels,
            n_upsamples=self.n_upsamples,
            x_dims=self.x_dims )
        # self.discriminator = Discriminator(
        #     dis_kernels=self.dis_kernels,
        #     soften_labels=self.soften_labels,
        #     soften_sddev=self.soften_sddev )

        ## ---------------------- Input ops ----------------------- ##
        if self.iterator_dataset:
            self.x_in = tf.placeholder(tf.float32,
                shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')
        else:
            self.x_in = tf.placeholder_with_default(self.dataset.image_op,
                shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')

        self.z_in_default = tf.random_normal([self.batch_size, self.z_dim],
            mean=0.0, stddev=1.0)
        self.z_in = tf.placeholder_with_default(self.z_in_default,
            shape=[None, self.z_dim])
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')

        ## ---------------------- Model ops ----------------------- ##
        self.zed, self.mu, self.log_var = self.encoder.model(self.x_in, keep_prob=self.keep_prob)
        self.x_hat = self.generator.model(self.zed, keep_prob=self.keep_prob)

        ## ---------------------- Loss ops ------------------------ ##
        self._loss_op()

        ## -------------------- Training ops ---------------------- ##
        self._training_ops()

        ## --------------------- Summary ops ---------------------- ##
        self._summary_ops()

        ## ------------------- TensorFlow ops --------------------- ##
        self._tf_ops()

        ## ---------------------- Initialize ---------------------- ##
        self._print_info_to_file(filename=os.path.join(self.save_dir,
            '{}_settings.txt'.format(self.name)))
        self.sess.run(tf.global_variables_initializer())



    def _loss_op(self):
        self.loss = tf.constant(0.0)
        self._reconstruction_loss()
        self._kl_divergence()

    def _reconstruction_loss(self):
        self.recon_loss = tf.losses.mean_squared_error(
            labels=self.x_in, predictions=self.x_hat)
        tf.assign_add(self.loss, self.recon_loss)

    def _kl_divergence(self):
        self.kld = -0.5 * tf.reduce_sum(1 + self.log_var
            - tf.square(self.mu)
            - tf.exp(self.log_var), 1)
        tf.assign_add(self.loss, self.kld)

    def _training_ops(self):
        self.generator_vars = self.generator.get_update_list()
        self.encoder_vars = self.encoder.get_update_list()

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def _summary_ops(self):
        self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=8)
        self.x_hat_sum = tf.summary.image('x_hat', self.x_hat, max_outputs=8)

        self.zed_sum = tf.summary.histogram('z', self.zed)
        self.mu_sum = tf.summary.histogram('mu', self.mu)
        self.logvar_sum = tf.summary.histogram('logvar', self.log_var)

        self.summary_op = tf.summary.merge_all()

    def train_step(self):
        self.global_step += 1
        self.sess.run(self.train_op)
        if self.global_step % self.summary_iters == 0:
            summary_str = self.sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.global_step)
