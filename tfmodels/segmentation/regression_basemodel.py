import tensorflow as tf
import numpy as np
import sys, os

from ..utilities.basemodel import BaseModel

class Regression(BaseModel):
    ## Defaults. arXiv links correspond to inspirational materials
    regression_defaults={
        'dataset': None,
        'global_step': 0,
        'k_size': 3,
        'learning_rate': 1e-3,
        'mode': 'TRAIN',
        'name': 'ImageRegression',
        'nonlin': tf.nn.selu,
        'log_dir': None,
        'save_dir': None,
        'sess': None,
        'reg_training_op_list': [],
        'summarize_grads': False,
        'summary_iters': 50,
        'summary_image_iters': 250,
        'summary_image_n': 4,
        'summary_op_list': [],
        'with_test': False,
        'n_test_batches': 10,
        'x_dims': [256, 256, 3],
     }

    def __init__(self, **kwargs):
        self.regression_defaults.update(**kwargs)

        super(Regression, self).__init__(**self.regression_defaults)
        assert self.sess is not None

        if self.mode=='TRAIN':
            self._training_mode()
        elif self.mode=='TEST':
            self._test_mode()


    def _training_mode(self):
        print 'Setting up {} in training mode'.format(self.name)
        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
            name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.mask_op,
            shape=[None, self.x_dims[0], self.x_dims[1], 1], name='y_in')

        ## Check for a testing dataset
        if self.dataset.testing_record is not None:
            self.with_test = True

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False)

        print 'Model output y_hat:', self.y_hat.get_shape()

        ## ------------------- Training ops ------------------- ##
        self.var_list = self._get_update_list()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
            name='{}_Adam'.format(self.name))

        self._make_training_ops()

        ## ------------------- Gather Summary ops ------------------- ##
        self._make_summaries()

        ## ------------------- Gather Testing ops ------------------- ##
        self._make_test_ops()

        ## ------------------- TensorFlow helpers ------------------- ##
        self._tf_ops()
        self.sess.run(tf.global_variables_initializer())

        self._print_info_to_file(filename=os.path.join(self.save_dir,
            '{}_settings.txt'.format(self.name)))


    def _test_mode(self):
        print 'Setting up {} in inference mode'.format(self.name)
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

        # self.make_saver() ## In progress (SAVE1)
        # with tf.device('/cpu:0'):
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())


    def _get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if self.name in var.name]


    ## define self.reg_loss
    def _make_regression_loss(self):
        # self.reg_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.y_hat,
        #     labels=self.y_in), )
        # self.reg_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.y_hat,
        #     labels=self.y_in))
        self.reg_loss = tf.losses.mean_squared_error( self.y_in, self.y_hat, )

    def _make_training_ops(self):
        with tf.name_scope('regression_losses'):
            self._make_regression_loss()

            ## Unused except in pretraining or specificially requested
            self.reg_training_op = self.optimizer.minimize(
                self.reg_loss, var_list=self.var_list, name='{}_reg_train'.format(self.name))

            self.reg_loss_sum = tf.summary.scalar('reg_loss', self.reg_loss)
            self.summary_op_list.append(self.reg_loss_sum)

            self.loss = self.reg_loss

            self.train_op = self.optimizer.minimize(
                self.loss, var_list=self.var_list, name='{}_train'.format(self.name))

            self.reg_training_op_list.append(self.train_op)


    def _make_summaries(self):
        ## https://github.com/aymericdamien/ \
        ## TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
        if self.summarize_grads:
            self.summary_gradient_list = []
            grads = tf.gradients(self.loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                self.summary_gradient_list.append(
                    tf.summary.histogram(var.name + '/gradient', grad))

        ## Loss scalar
        with tf.variable_scope('training_scalars'):
            self.loss_sum = tf.summary.scalar('loss', self.loss)
            self.y_in_hist = tf.summary.histogram('y_in_hist', self.y_in)
            self.y_hat_hist = tf.summary.histogram('y_hat_hist', self.y_hat)

            ## Scalars:
            self.summary_scalars_op = tf.summary.merge_all()

        ## Images
        with tf.variable_scope('training_images'):
            self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=self.summary_image_n)
            self.y_in_sum = tf.summary.image('y_in', self.y_in, max_outputs=self.summary_image_n)
            self.y_hat_sum = tf.summary.image('y_hat', self.y_hat, max_outputs=self.summary_image_n)

        self.summary_images_op = tf.summary.merge(
            [self.x_in_sum, self.y_in_sum, self.y_hat_sum])


    def _make_test_ops(self):
        if self.with_test is None:
            print 'WARNING no TEST tfrecord dataset; Skipping test mode'
            return

        with tf.variable_scope('testing_scalars'):
            self.loss_sum_test = tf.summary.scalar('loss_test', self.loss)

        with tf.variable_scope('testing_images'):
            self.x_in_sum_test = tf.summary.image('x_in_test', self.x_in, max_outputs=self.summary_image_n)
            self.y_in_sum_test = tf.summary.image('y_in_test', self.y_in, max_outputs=self.summary_image_n)
            self.y_hat_sum_test = tf.summary.image('y_hat_test', self.y_hat, max_outputs=self.summary_image_n)

            self.summary_test_ops = tf.summary.merge(
                [self.loss_sum_test, self.x_in_sum_test,
                 self.y_in_sum_test, self.y_hat_sum_test])


    def _write_scalar_summaries(self):
        summary_str, reg_loss_ = self.sess.run([self.summary_scalars_op, self.reg_loss])
        self.summary_writer.add_summary(summary_str, self.global_step)
        print '[{:07d}] writing scalar summaries (loss={:3.3f})'.format(self.global_step, reg_loss_)


    def _write_image_summaries(self):
        print '[{:07d}] writing image summaries'.format(self.global_step)
        summary_str = self.sess.run(self.summary_images_op)
        self.summary_writer.add_summary(summary_str, self.global_step)


    def inference(self, x_in, keep_prob=1.0):
        feed_dict = {self.x_in: x_in,
                     self.keep_prob: keep_prob}
        y_hat_ = self.sess.run([self.y_hat], feed_dict=feed_dict)[0]
        # y_hat_ = self.sess.run([self.y_hat], feed_dict=feed_dict)[0]
        return y_hat_


    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)


    def test_step(self, step_delta, keep_prob=0.8):
        fd = {self.keep_prob: keep_prob}
        summary_str, test_loss_ = self.sess.run([self.summary_test_ops, self.loss], feed_dict=fd)
        self.summary_writer.add_summary(summary_str, self.global_step+step_delta)
        print '#### TEST #### [{:07d}] writing test summaries (loss={:3.3f})'.format(self.global_step, test_loss_)


    def train_step(self):
        self.global_step += 1
        self.sess.run(self.reg_training_op_list)

        if self.global_step % self.summary_iters == 0:
            self._write_scalar_summaries()

        if self.global_step % self.summary_image_iters == 0:
            self._write_image_summaries()

    """ Run a number of testing iterations """
    def test(self):
        ## Switch dataset to testing
        self.dataset._initalize_testing(self.sess)

        for step_delta in xrange(self.n_test_batches):
            self.test_step(step_delta)

        self.dataset._initalize_training(self.sess)
