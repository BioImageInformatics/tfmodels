import tensorflow as tf
import numpy as np
import sys, os

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from basemodel import BaseModel
from discriminator import ConvDiscriminator

from utilities.ops import (
    lrelu,
    linear,
    conv,
    deconv,
    batch_norm)

class GenericSegmentation(BaseModel):
    defaults={
        'sess': None,
        'learning_rate': 1e-3,
        'adversarial': False,
        'dataset': None,
        # 'x_size': [256, 256],
        'conv_kernels': [32, 64],
        'deconv_kernels': [16, 16, 32],
        'n_classes': None,
        'mode': 'TRAIN', }

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        super(GenericSegmentation, self).__init__(**self.defaults)

        assert self.n_classes is not None
        if self.mode=='TRAIN': assert self.dataset.dstype=='ImageMask'

        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]], name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.mask_op,
            shape=[None, self.x_dims[0], self.x_dims[1], 1], name='y_in')
        # self.x_in = self.dataset.image_op
        # self.y_in = self.dataset.mask_op
        if self.y_in.get_shape().as_list()[-1] != self.n_classes:
            self.y_in = tf.one_hot(self.y_in, depth=self.n_classes)
            self.y_in = tf.squeeze(self.y_in)
            print 'Converted y_in to one_hot: ', self.y_in.get_shape()

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False, training=True)
        print 'Model output y_hat:', self.y_hat.get_shape()
        if self.adversarial:
            self.discriminator = ConvDiscriminator(sess=self.sess,
                x_real=self.y_in, x_fake=self.y_hat)
            self.discriminator.print_info()
            self.training_op_list += self.discriminator.training_op_list
            self.summary_op_list += self.discriminator.summary_op_list

        self.loss = self.loss_op()

        ## ------------------- Training ops ------------------- ##
        self.var_list = self.get_update_list()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='GenSegAdam')
        self.training_op = self.optimizer.minimize(self.loss, var_list=self.var_list)
        self.training_op_list.append(self.training_op)

        ## ------------------- Gather Summary ops ------------------- ##
        self.summary_op_list += self.summaries()
        self.summary_op = tf.summary.merge(self.summary_op_list)
        self.training_op_list.append(self.summary_op)

        ## ------------------- TensorFlow helpers ------------------- ##
        self.summary_writer = tf.summary.FileWriter(self.log_dir,
            graph=self.sess.graph, flush_secs=30)
        ## Append a model name to the save path
        self.save_dir = os.path.join(self.save_dir, 'generic_segmentation.ckpt')
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())


    def get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if 'GenericSeg' in var.name]

    def summaries(self):
        x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=4)

        y_in_mask = tf.expand_dims(tf.argmax(self.y_in, -1),-1)
        y_in_mask = tf.cast(y_in_mask, tf.float32)
        y_in_sum = tf.summary.image('y_in', y_in_mask, max_outputs=4)
        print '\t y_in_mask', y_in_mask.get_shape(), y_in_mask.dtype

        y_hat_mask = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
        y_hat_mask = tf.cast(y_hat_mask, tf.float32)
        y_hat_sum = tf.summary.image('y_hat', y_hat_mask, max_outputs=4)
        print '\t y_hat_mask', y_hat_mask.get_shape(), y_hat_mask.dtype

        loss_sum = tf.summary.scalar('loss', self.loss)

        return [x_in_sum, y_in_sum, y_hat_sum, loss_sum]


    def train_step(self, global_step):
        summary_str = self.sess.run(self.training_op_list)[-1]
        self.summary_writer.add_summary(summary_str, global_step)


    def snapshot(self, step):
        print 'Snapshotting to [{}] step [{}]'.format(self.save_dir, step)
        self.saver.save(self.sess, self.save_dir, global_step=step)


    def restore(self, snapshot_path):
        print 'Restoring from {}'.format(snapshot_path)
        try:
            self.saver.restore(self.sess, snapshot_path)
            print 'Success!'
        except:
            print 'Failed! Continuing without loading snapshot.'


    def test_step(self, keep_prob=1.0):
        x_in_, y_in_, y_hat_ = self.sess.run([self.x_in, self.y_in, self.y_hat],
            feed_dict={self.keep_prob: keep_prob})
        return [x_in_, y_in_, y_hat_]


    def inference(self, x_in, keep_prob):
        y_hat_ = self.sess.run([self.y_hat],
            feed_dict={self.x_in: x_in, self.keep_prob: keep_prob})[0]

        return y_hat_



    def loss_op(self):
        seg_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_in, logits=self.y_hat)
        seg_loss = tf.reduce_mean(seg_loss)
        seg_loss_sum = tf.summary.scalar('seg_loss', seg_loss)
        self.summary_op_list.append(seg_loss_sum)

        if self.adversarial:
            p_real_fake = self.discriminator.p_real_fake
            real_target = tf.ones_like(p_real_fake)
            adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=real_target, logits=p_real_fake))
            adv_loss_sum = tf.summary.scalar('adv_loss', adv_loss)
            self.summary_op_list.append(adv_loss_sum)
            return seg_loss + adv_loss

        return seg_loss


    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'GenericSegmentation Model'
        with tf.variable_scope('GenericSeg') as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            c0 = conv(x_in, self.conv_kernels[0], var_scope='c0')
            c0 = tf.nn.dropout(c0, keep_prob=keep_prob)
            c0 = batch_norm(c0, training=training, var_scope='c0_bn')
            c0 = lrelu(c0)
            c0_pool = tf.nn.max_pool(c0, [1,2,2,1], [1,2,2,1], padding='VALID', name='c0_pool')
            print '\t c0_pool', c0_pool.get_shape()

            c1 = conv(c0_pool, self.conv_kernels[1], var_scope='c1')
            c1 = tf.nn.dropout(c1, keep_prob=keep_prob)
            c1 = batch_norm(c1, training=training, var_scope='c1_bn')
            c1 = lrelu(c1)
            c1_pool = tf.nn.max_pool(c1, [1,2,2,1], [1,2,2,1], padding='VALID', name='c1_pool')
            print '\t c1_pool', c1_pool.get_shape()

            d2 = deconv(c1_pool, self.deconv_kernels[2], var_scope='d2')
            d2 = tf.nn.dropout(d2, keep_prob=keep_prob)
            d2 = batch_norm(d2, training=training, var_scope='d2_bn')
            d2 = lrelu(d2)
            print '\t d2', d2.get_shape()

            d1 = deconv(d2, self.deconv_kernels[1], var_scope='d1')
            d1 = tf.nn.dropout(d1, keep_prob=keep_prob)
            d1 = batch_norm(d1, training=training, var_scope='d1_bn')
            d1 = lrelu(d1)
            print '\t d1', d1.get_shape()

            d0 = deconv(d1, self.deconv_kernels[0], var_scope='d0')
            d0 = tf.nn.dropout(d0, keep_prob=keep_prob)
            d0 = batch_norm(d0, training=training, var_scope='d0_bn')
            d0 = lrelu(d0)
            print '\t d0', d0.get_shape()

            y_hat = deconv(d0, self.n_classes, var_scope='y_hat')
            print '\t y_hat', y_hat.get_shape()

            return y_hat


    def print_info(self):
        print '------------------------ GenericSegmentation ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ GenericSegmentation ---------------------- '
