import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '.')
from basemodel import BaseModel
sys.path.insert(0, '../utilities')
from ops import (
    lrelu,
    linear,
    conv,
    batch_norm)

class ConvDiscriminator(BaseModel):
    defaults={
        x_real: None,
        x_fake: None,
        learning_rate: 1e-4,
        kernels: [32, 64, 512],
        name: 'ConvDiscriminator'}

    def __init__(self, **kwargs):
        defaults.update(kwargs)
        super(ConvDiscriminator, self).__init__(**defaults)

        assert self.x_real is not None
        assert self.x_fake is not None

        self.p_real_fake = self.model(self.x_fake)
        self.p_real_real = self.model(self.x_real, reuse=True)
        self.loss = self.loss_op()

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)

        self.training_op_list.append(self.training_op)

        discrim_loss_sum = tf.summary.scalar('discrim_loss', self.loss)
        self.summary_op_list.append(discrim_loss_sum)


    def loss_op(self):
        real_target = tf.ones_like(self.p_real_fake)
        fake_target = tf.zeros_like(self.p_real_real)

        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=real_target, logits=self.p_real_real))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=fake_target, logits=self.p_real_fake))
        return (loss_real + loss_fake) / 2.0


    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        print 'Convolutional Discriminator'
        with tf.variable_scope('ConvDiscriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h0 = conv(x_hat, self.kernels[0], var_scope='h0')
            h0 = batch_norm(h0, training=training, reuse=reuse, var_scope='h0_bn')
            h0 = lrelu(h0)

            h1 = conv(h0, self.kernels[1], var_scope='h1')
            h1 = batch_norm(h1, training=training, reuse=reuse, var_scope='h1_bn')
            h1 = lrelu(h1)

            h1_flat = tf.contrib.layers.flatten(h1)
            h2 = lrelu(linear(h1_flat, self.kernels[2], var_scope='h2'))
            h2 = tf.nn.dropout(h2, keep_prob=keep_prob, name='h2_do')

            p_real = linear(h2, 1, name='p_real')

            return p_real
