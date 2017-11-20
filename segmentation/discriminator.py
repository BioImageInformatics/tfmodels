import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '.')
from basemodel import BaseModel
from ops import (
    lrelu,
    linear,
    conv,
    batch_norm)

class ConvDiscriminator(BaseModel):
    defaults={
        'x_real': None,
        'x_fake': None,
        'learning_rate': 5e-4,
        'kernels': [16, 64, 512],
        'real_softening': 0.1,
        'name': 'ConvDiscriminator'}

    def __init__(self, **kwargs):
        self.defaults.update(kwargs)
        super(ConvDiscriminator, self).__init__(**self.defaults)

        assert self.x_real is not None
        assert self.x_fake is not None

        ## Label softening add noise ~ N(0,0.01)
        epsilon = tf.random_normal(shape=tf.shape(self.x_real),
            mean=0.0, stddev=self.real_softening)
        self.x_real = self.x_real + epsilon

        self.p_real_fake = self.model(self.x_fake)
        self.p_real_real = self.model(self.x_real, reuse=True)
        self.loss = self.loss_op()

        self.var_list = self.get_update_list()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='DiscAdam')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.training_op = self.optimizer.minimize(self.loss, var_list=self.var_list)

        self.training_op_list.append(self.training_op)

        discrim_loss_sum = tf.summary.scalar('discrim_loss', self.loss)
        self.summary_op_list.append(discrim_loss_sum)


    def get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if 'ConvDiscriminator' in var.name]


    ## TODO switch to Wasserstein loss. Remember to clip the outputs
    def loss_op(self):
        real_target = tf.ones_like(self.p_real_fake)
        fake_target = tf.zeros_like(self.p_real_real)

        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=real_target, logits=self.p_real_real))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=fake_target, logits=self.p_real_fake))
        return (loss_real + loss_fake) / 2.0


    def model(self, y_hat, keep_prob=0.5, reuse=False, training=True):
        print 'Convolutional Discriminator'
        with tf.variable_scope('ConvDiscriminator') as scope:
            if reuse:
                scope.reuse_variables()
            print '\t y_hat', y_hat.get_shape()

            h0 = conv(y_hat, self.kernels[0], k_size=7, stride=4, var_scope='h0')
            h0 = batch_norm(h0, training=training, var_scope='h0_bn')
            h0 = lrelu(h0)

            h0_pool = tf.nn.max_pool(h0, [1,3,3,1], [1,3,3,1], padding='VALID',
                name='h0_pool')
            print '\t h0_pool', h0_pool.get_shape()

            h1 = conv(h0_pool, self.kernels[1], var_scope='h1')
            h1 = batch_norm(h1, training=training, var_scope='h1_bn')
            h1 = lrelu(h1)

            h1_pool = tf.nn.max_pool(h1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='h1_pool')
            print '\t h1_pool', h1_pool.get_shape()

            h1_flat = tf.contrib.layers.flatten(h1_pool)
            h1_flat = tf.nn.dropout(h1_flat, keep_prob=keep_prob, name='h1_flat_do')
            print '\t h1_flat', h1_flat.get_shape()

            h2 = lrelu(linear(h1_flat, self.kernels[2], var_scope='h2'))
            h2 = tf.nn.dropout(h2, keep_prob=keep_prob, name='h2_do')
            print '\t h2', h2.get_shape()

            p_real = linear(h2, 1, var_scope='p_real')
            print '\t p_real', p_real.get_shape()

            return p_real

    def inference(self, x_in, keep_prob=1.0):
        p_real_ = self.sess.run([p_real_fake], feed_dict={self.x_fake: x_in})
        p_real_smax = tf.nn.softmax(p_real_)
        return p_real_smax


    def print_info(self):
        print '------------------------ ConvDiscriminator ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ ConvDiscriminator ---------------------- '
