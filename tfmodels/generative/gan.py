from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

from ..utilities.basemodel import BaseModel
from .discriminator_basemodel import BaseDiscriminator
from .generator_basemodel import BaseGenerator
from ..utilities.ops import (
    conv,
    deconv,
    linear,
    conv_cond_concat
)

""" Generative Adversarial Network

1. Define the disciminator and generator according to the template models
2. implement the loss functions and regularizers
3. ???

"""
class Discriminator(BaseDiscriminator):
    gan_discriminator_defaults = {
        'dis_kernels': [32, 64, 128]
    }

    def __init__(self, **kwargs):
        self.gan_discriminator_defaults.update(**kwargs)
        super(Discriminator, self).__init__(**self.gan_discriminator_defaults)


    def model(self, x_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print('Setting up GAN/Discriminator')
            print('Nonlinearity: ', self.nonlin)
            nonlin = self.nonlin

            print('\t x_in', x_in.get_shape())
            c0 = nonlin(conv(x_in, self.dis_kernels[0], k_size=5, stride=3, var_scope='c0', selu=1))
            c1 = nonlin(conv(c0, self.dis_kernels[1], k_size=5, stride=3, var_scope='c1', selu=1))
            # c2 = nonlin(conv(c1, self.dis_kernels[1], k_size=5, stride=3, var_scope='c2'))
            flat = tf.contrib.layers.flatten(c1)
            print('\t flat', flat.get_shape())
            h0 = nonlin(linear(flat, self.dis_kernels[2], var_scope='h0', selu=1))
            p_real = linear(h0, 1, var_scope='p_real', no_bias=True)
            print('\t p_real', p_real.get_shape())

            return tf.nn.sigmoid(p_real)



class Generator(BaseGenerator):
    gan_generator_defaults = {
        'gen_kernels': [128, 64, 32],
        'x_dims': [128, 128, 3],
    }

    def __init__(self, **kwargs):
        self.gan_generator_defaults.update(**kwargs)
        super(Generator, self).__init__(**self.gan_generator_defaults)

    def model(self, z_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print('Setting up GAN/Generator')
            print('Nonlinearity: ', self.nonlin)
            nonlin = self.nonlin

            ## Project
            print('\t z_in', z_in.get_shape())
            projection = nonlin(linear(z_in, self.project_shape, var_scope='projection', selu=1))
            project_conv = tf.reshape(projection, self.resize_shape)
            print('\t project_conv', project_conv.get_shape())
            h0 = nonlin(deconv(project_conv, self.gen_kernels[0], var_scope='h0', selu=1))
            h1 = nonlin(deconv(h0, self.gen_kernels[1], var_scope='h1', selu=1))

            x_hat = conv(h1, self.x_dims[-1], stride=1, var_scope='x_hat')
            print('\t x_hat', x_hat.get_shape())

            return x_hat



## defaults for all variables needed in Generator and Discriminator
class GAN(BaseModel):
    gan_defaults = {
        'batch_size': 64,
        'dataset': None,
        'discriminator': None,
        'dis_learning_rate': 1e-4,
        'dis_kernels': [32, 64, 128, 256],
        'generator': None,
        'gen_learning_rate': 1e-4,
        'gen_kernels': [32, 64, 128, 256],
        'global_step': 0,
        'iterator_dataset': False,
        'log_dir': None,
        'mode': 'TRAIN',
        'name': 'GAN',
        'pretraining': None,
        'save_dir': None,
        'sess': None,
        'soften_labels': False,
        'soften_sddev': 0.01,
        'summary_iters': 50,
        'summarize_grads': False,
        'x_dims': [256, 256, 3],
        'z_dim': 64, }

    def __init__(self, **kwargs):
        self.gan_defaults.update(**kwargs)
        super(GAN, self).__init__(**self.gan_defaults)

        assert self.sess is not None
        assert len(self.x_dims) == 3
        if self.mode=='TRAIN': assert self.dataset is not None

        self.generator = Generator(
            gen_kernels=self.gen_kernels,
            x_dims=self.x_dims )
        self.discriminator = Discriminator(
            dis_kernels=self.dis_kernels,
            soften_labels=self.soften_labels,
            soften_sddev=self.soften_sddev )


        ## ---------------------- Input ops ----------------------- ##
        if self.iterator_dataset:
            self.x_in = tf.placeholder(tf.float32,
                shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')
        else:
            self.x_in = tf.placeholder_with_default(self.dataset.image_op,
                shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')

        self.zed_default = tf.random_normal([self.batch_size, self.z_dim],
            mean=0.0, stddev=1.0)
        self.zed = tf.placeholder_with_default(self.zed_default,
            shape=[None, self.z_dim], name='zed')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')

        ## ---------------------- Model ops ----------------------- ##
        self.x_hat = self.generator.model(self.zed, keep_prob=self.keep_prob)
        self.p_real_real = self.discriminator.model(self.x_in, keep_prob=self.keep_prob)
        self.p_real_fake = self.discriminator.model(self.x_hat, keep_prob=self.keep_prob, reuse=True)

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

        ## ---------------------- Pretraining --------------------- ##
        if self.pretraining is not None:
            self._pretraining()


    def _loss_op(self):
        ## Define losses
        # self.real_target = tf.ones_like(self.p_real_real)
        # self.fake_target = tf.zeros_like(self.p_real_fake)
        #
        # if self.soften_labels:
        #     real_epsilon = tf.random_normal(shape=tf.shape(real_target),
        #         mean=0.0, stddev=self.soften_sddev)
        #     fake_epsilon = tf.random_normal(shape=tf.shape(fake_target),
        #         mean=0.0, stddev=self.soften_sddev)
        #     self.real_target = self.real_target + real_epsilon
        #     self.fake_target = self.fake_target + fake_epsilon

        # self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=self.real_target, logits=self.p_real_real ))
        # self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=self.fake_target, logits=self.p_real_fake ))

        # self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=self.real_target, logits=self.p_real_fake ))
        #
        # self.discriminator_loss = self.dis_loss_real + self.dis_loss_fake

        TINY = 1e-11
        self.generator_loss = -tf.reduce_mean(tf.log(self.p_real_fake+TINY))
        self.discriminator_loss = -tf.reduce_mean(tf.log(self.p_real_real+TINY) + tf.log(1. - self.p_real_fake+TINY))

        self.generator_loss_sum = tf.summary.scalar('generator_loss', self.generator_loss)
        self.discriminator_loss_sum = tf.summary.scalar('discriminator_loss', self.discriminator_loss)


    def _training_ops(self):
        ## Define training ops
        # self.generator_vars = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        # self.discriminator_vars = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        self.generator_vars = self.generator.get_update_list()
        self.discriminator_vars = self.discriminator.get_update_list()

        self.generator_optimizer = tf.train.AdamOptimizer(self.gen_learning_rate)
        self.discriminator_optimizer = tf.train.AdamOptimizer(self.dis_learning_rate)

        self.gen_train_op = self.generator_optimizer.minimize(self.generator_loss, var_list=self.generator_vars)
        self.dis_train_op = self.discriminator_optimizer.minimize(self.discriminator_loss, var_list=self.discriminator_vars)

        self.gan_train_op_list = [self.gen_train_op, self.dis_train_op]
        # self.training_op_list.append(self.gen_train_op)
        # self.training_op_list.append(self.dis_train_op)

    def _summary_ops(self):
        if self.summarize_grads:
            self.summary_gradient_list = []
            grads = tf.gradients(self.generator_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                self.summary_gradient_list.append(
                    tf.summary.histogram(var.name + '/gradient', grad))

        self.x_hat_sum = tf.summary.image('x_hat', self.x_hat, max_outputs=8)
        self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=8)

        self.summary_op = tf.summary.merge_all()
        # self.training_op_list.append(self.summary_op)


    def _pretraining(self):
        print('Pretraining discriminator')
        for _ in xrange(self.pretraining):
            if self.iterator_dataset:
                feed_dict = {self.x_in: next(self.dataset.iterator)}
                _ = self.sess.run([self.dis_train_op], feed_dict=feed_dict)
            else:
                _ = self.sess.run([self.dis_train_op])

        print('Pretraining generator')
        for _ in xrange(self.pretraining):
            if self.iterator_dataset:
                feed_dict = {self.x_in: next(self.dataset.iterator)}
                _ = self.sess.run([self.gen_train_op], feed_dict=feed_dict)
            else:
                _ = self.sess.run([self.gen_train_op])

    def inference(self, z_values):
        ## Take in values for z and return p(data|z)
        feed_dict = {self.zed: z_values}
        x_hat = self.sess.run(self.x_hat, feed_dict=feed_dict)
        return x_hat

    def train_step(self):
        self.global_step += 1
        # if self.iterator_dataset:
        feed_dict = {self.x_in: next(self.dataset.iterator)}
        self.sess.run(self.gan_train_op_list, feed_dict=feed_dict)
        # else:
        #     self.sess.run(self.train_op)

        if self.global_step % self.summary_iters == 0:
            summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_str, self.global_step)
