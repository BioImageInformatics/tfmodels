import tensorflow as tf
import numpy as np
import sys, os

# sys.path.insert(0, '.')
from basemodel import BaseModel
from discriminator import ConvDiscriminator
from ops import (
    lrelu,
    linear,
    conv,
    deconv,
    batch_norm)

class FCNBase(BaseModel):
    base_defaults={
        'sess': None,
        'learning_rate': 1e-3,
        'adversarial': False,
        'adversary_lr': 1e-4,
        'adversary_lambda': 1,
        'dataset': None,
        'x_dims': [256, 256, 3],
        'conv_kernels': [32, 64, 128, 256],
        'deconv_kernels': [64],
        'n_classes': None,
        'summary_iters': 50,
        'mode': 'TRAIN',
        'name': 'SegNet'}

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        super(FCNBase, self).__init__(**self.base_defaults)

        assert self.n_classes is not None
        if self.mode=='TRAIN': assert self.dataset.dstype=='ImageMask'


    ## Layer flow copied from:
    ## https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py
    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'FCN Model'
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=3, stride=1, var_scope='c0_0'))
            c0_1 = nonlin(conv(c0_0, self.conv_kernels[0], k_size=3, stride=1, var_scope='c0_1'))
            c0_pool = tf.nn.max_pool(c0_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c0_pool')
            print '\t c0_pool', c0_pool.get_shape() ## 128

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=3, stride=1, var_scope='c1_0'))
            c1_1 = nonlin(conv(c1_0, self.conv_kernels[1], k_size=3, stride=1, var_scope='c1_1'))
            c1_pool = tf.nn.max_pool(c1_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c1_pool')
            print '\t c1_pool', c1_pool.get_shape() ## 64

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=3, stride=1, var_scope='c2_0'))
            c2_1 = nonlin(conv(c2_0, self.conv_kernels[2], k_size=3, stride=1, var_scope='c2_1'))
            c2_pool = tf.nn.max_pool(c2_1, [1,4,4,1], [1,4,4,1], padding='VALID',
                name='c2_pool')
            print '\t c2_pool', c2_pool.get_shape() ## 32

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=3, stride=1, var_scope='c3_0'))
            c3_0 = tf.contrib.nn.alpha_dropout(c3_0, keep_prob=keep_prob)
            c3_1 = nonlin(conv(c3_0, self.conv_kernels[3], k_size=3, stride=1, var_scope='c3_1'))
            c3_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c3_pool')
            print '\t c3_pool', c3_pool.get_shape()  ## inputs / 16 = 16

            ## The actual ones has one more instead
            # c4_0 = nonlin(conv(c3_pool, self.conv_kernels[4], k_size=3, stride=1, var_scope='c4_0'))
            # c4_0 = tf.contrib.nn.alpha_dropout(c4_0, keep_prob=keep_prob)
            # c4_1 = nonlin(conv(c4_0, self.conv_kernels[4], k_size=3, stride=1, var_scope='c4_1'))
            # c4_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
            #     name='c4_pool')
            # print '\t c4_pool', c4_pool.get_shape()  ## inputs / 32 = 8

            upscore3 = nonlin(deconv(c3_pool, self.n_classes, upsample_rate=16, var_scope='ups3'))
            upscore2 = nonlin(deconv(c2_pool, self.n_classes, upsample_rate=8, var_scope='ups2'))
            upscore1 = nonlin(deconv(c1_pool, self.n_classes, upsample_rate=4, var_scope='ups1'))
            print '\t upscore3', upscore3.get_shape()
            print '\t upscore2', upscore2.get_shape()
            print '\t upscore1', upscore1.get_shape()

            upscore_concat = tf.concat([upscore3, upscore2, upscore1], axis=-1)
            print '\t upscore_concat', upscore_concat.get_shape()
            d0 = nonlin(conv(upscore_concat, deconv_kernels[0], k_size=3, stride=1, var_scope='d0'))
            print '\t d0', d0.get_shape()

            y_hat = deconv(d0, self.n_classes, var_scope='y_hat')
            print '\t y_hat', y_hat.get_shape() ## 128*2 = 256

            return y_hat




class FCNTraining(FCNBase):
    train_defaults = {
    'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(FCNTraining, self).__init__(**self.train_defaults)

        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.mask_op,
            shape=[None, self.x_dims[0], self.x_dims[1], 1], name='y_in')
        # self.x_in = self.dataset.image_op
        # self.y_in = self.dataset.mask_op
        if self.y_in.get_shape().as_list()[-1] != self.n_classes:
            self.y_in_mask = tf.cast(tf.identity(self.y_in), tf.float32)
            # self.y_in_mask = tf.divide(self.y_in_mask, self.n_classes)
            self.y_in = tf.one_hot(self.y_in, depth=self.n_classes)
            self.y_in = tf.squeeze(self.y_in)
            self.y_in = tf.reshape(self.y_in,
                [-1, self.x_dims[0], self.x_dims[1], self.n_classes])
            print 'Converted y_in to one_hot: ', self.y_in.get_shape()

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.training = tf.placeholder_with_default(True, shape=[], name='training')
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False,
            training=self.training)
        self.y_hat_smax = tf.nn.softmax(self.y_hat)
        self.y_hat_mask = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
        self.y_hat_mask = tf.cast(self.y_hat_mask, tf.float32)
        # self.y_hat_mask = tf.divide(self.y_hat_mask, self.n_classes)
        print 'Model output y_hat:', self.y_hat.get_shape()

        ## ------------------- Training ops ------------------- ##
        self.var_list = self.get_update_list()
        self.seg_optimizer = tf.train.AdamOptimizer(self.learning_rate, name='FCN_Adam')

        if self.adversarial:
            # self.adv_optimizer = tf.train.AdamOptimizer(self.adversary_lr, name='VGG_adv_Adam')
            self.discriminator = ConvDiscriminator(sess=self.sess,
                x_in=self.x_in, y_real=self.y_in, y_fake=tf.nn.softmax(self.y_hat))
            # self.discriminator = ConvDiscriminator(sess=self.sess,
            #     x_real=self.y_in_mask, x_fake=self.y_hat_mask)
            self.discriminator.print_info()
            self.training_op_list += self.discriminator.training_op_list
            self.summary_op_list += self.discriminator.summary_op_list

        self.make_training_ops()

        ## ------------------- Gather Summary ops ------------------- ##
        self.summary_op_list += self.summaries()
        # self.summary_op = tf.summary.merge(self.summary_op_list)
        self.summary_op = tf.summary.merge_all()
        self.training_op_list.append(self.summary_op)

        ## ------------------- TensorFlow helpers ------------------- ##
        self.summary_writer = tf.summary.FileWriter(self.log_dir,
            graph=self.sess.graph, flush_secs=30)
        ## Append a model name to the save path
        self.snapshot_name = os.path.join(self.save_dir, 'fcn_segmentation.ckpt')
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

        self._print_settings(filename=os.path.join(self.save_dir, 'fcn_model_settings.txt'))

    def make_training_ops(self):
        self.seg_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_in, logits=self.y_hat)
        self.seg_loss = tf.reduce_mean(self.seg_loss)

        self.seg_loss_sum = tf.summary.scalar('seg_loss', self.seg_loss)
        self.summary_op_list.append(self.seg_loss_sum)

        ## For batch norm
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.segmentation_train_op = self.seg_optimizer.minimize(
        #         self.seg_loss, var_list=self.var_list)
        #
        # self.training_op_list.append(self.segmentation_train_op)

        if self.adversarial:
            # p_real_fake = tf.stop_gradient(self.discriminator.model(self.y_hat_mask, reuse=True))
            p_real_fake = self.discriminator.p_real_fake
            real_target = tf.ones_like(p_real_fake)
            real_epsilon = tf.random_normal(shape=tf.shape(real_target),
                mean=0.0, stddev=0.01)
            real_target = real_target + real_epsilon
            self.adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=real_target, logits=p_real_fake)
            self.adv_loss = tf.reduce_mean(self.adv_loss)
            self.adv_loss_sum = tf.summary.scalar('adv_loss', self.adv_loss)
            self.summary_op_list.append(self.adv_loss_sum)

            # self.adversarial_train_op = self.adv_optimizer.minimize(
            #     self.adv_loss, var_list=self.var_list)
            # self.training_op_list.append(self.adversarial_train_op)
            self.loss = self.seg_loss + self.adversary_lambda * self.adv_loss
        else:
            self.loss = self.seg_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.training_op = self.seg_optimizer.minimize(
                self.loss, var_list=self.var_list)

        self.training_op_list.append(self.training_op)


    def get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if self.name in var.name]

    def summaries(self):
        ## Input image
        x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=4)
        y_in_sum = tf.summary.image('y_in', self.y_in_mask, max_outputs=4)
        y_hat_sum = tf.summary.image('y_hat', self.y_hat_mask, max_outputs=4)
        ## Loss scalar
        loss_sum = tf.summary.scalar('loss', self.loss)
        ## Filters
        # TODO

        return [x_in_sum, y_in_sum, y_hat_sum, loss_sum]

    def train_step(self, global_step):
        summary_str = self.sess.run(self.training_op_list)[-1]
        if global_step % self.summary_iters == 0:
            self.summary_writer.add_summary(summary_str, global_step)

    def train_step_return_values(self, global_step):
        train_return_ = self.sess.run(self.training_op_list+[self.x_in, self.y_in, self.y_hat_mask])
        return_x = train_return_[-3]
        return_y = train_return_[-2]
        return_y_hat = train_return_[-1]
        if global_step % self.summary_iters == 0:
            summary_str = train_return_[-4]
            self.summary_writer.add_summary(summary_str, global_step)

        return return_x, return_y, return_y_hat

    def snapshot(self, step):
        print 'Snapshotting to [{}] step [{}]'.format(self.snapshot_name, step),
        self.saver.save(self.sess, self.snapshot_name, global_step=step)
        print 'Done'

    def restore(self, snapshot_path):
        print 'Restoring from {}'.format(snapshot_path)
        try:
            self.saver.restore(self.sess, snapshot_path)
            print 'Success!'
        except:
            print 'Failed! Continuing without loading snapshot.'

    def inference(self, x_in, keep_prob):
        feed_dict = {self.x_in: x_in,
                     self.keep_prob: keep_prob,
                     self.training: False}
        y_hat_ = self.sess.run([self.y_hat_smax], feed_dict=feed_dict)[0]
        # y_hat_smax = tf.nn.softmax(y_hat_)

        return y_hat_


class FCNInference(FCNBase):
    inference_defaults = {
        'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(FCNInference, self).__init__(**self.inference_defaults)

        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder('float',
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
            name='x_in')

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.training = tf.placeholder_with_default(False, shape=[], name='training')
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False,
            training=self.training)
        self.y_hat_smax = tf.nn.softmax(self.y_hat)

        # self.y_hat_mask = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
        # self.y_hat_mask = tf.cast(self.y_hat_mask, tf.float32)

        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, x_in, keep_prob=1.0):
        assert len(x_in.shape) == 4
        feed_dict = {self.x_in: x_in,
                     self.keep_prob: keep_prob}
        y_hat_ = self.sess.run([self.y_hat_smax], feed_dict=feed_dict)[0]
        # y_hat_ = self.sess.run([self.y_hat], feed_dict=feed_dict)[0]

        return y_hat_

    def bayesian_inference(self, x_in, samples=25, keep_prob=0.5):
        assert keep_prob < 1.0
        assert samples > 1
        assert x_in.shape[0] == 1 and len(x_in.shape) == 4

        y_hat_ = self.inference(x_in=x_in, keep_prob=keep_prob)
        y_hat_ = np.expand_dims(y_hat_, -1)
        for tt in xrange(1, samples):
            y_hat_p = self.inference(x_in=x_in, keep_prob=keep_prob)
            y_hat_ = np.concatenate([y_hat_, np.expand_dims(y_hat_p, -1)], -1)

        y_bar_mean = np.mean(y_hat_, axis=-1) ## (1, h, w, n_classes)
        y_bar_var = np.var(y_hat_, axis=-1)
        y_bar = np.argmax(y_bar_mean, axis=-1) ## (1, h, w)

        return y_bar_mean, y_bar_var, y_bar


    def restore(self, snapshot_path):
        print 'Restoring from {}'.format(snapshot_path)
        try:
            self.saver.restore(self.sess, snapshot_path)
            print 'Success!'
        except:
            print 'Failed! Continuing without loading snapshot.'
