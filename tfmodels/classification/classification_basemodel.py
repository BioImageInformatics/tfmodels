import tensorflow as tf
import numpy as np
import sys, os

from ..utilities.basemodel import BaseModel

class Classifier(BaseModel):
    ## Defaults. arXiv links correspond to inspirational materials
    classification_defaults={
        'class_weights': None, ## https://arxiv.org/abs/1511.00561
        'dataset': None,
        'global_step': 0,
        'k_size': 3,
        'learning_rate': 1e-3,
        'log_dir': None,
        'mode': 'TRAIN',
        'name': 'Classifier',
        'nonlin': tf.nn.selu,
        'n_classes': None,
        'save_dir': None,
        'sess': None,
        'class_training_op_list': [],
        'summarize_grads': False,
        'summary_iters': 10,
        # 'summary_image_iters': 250,
        # 'summary_image_n': 4,
        'summary_op_list': [],
        'with_test': False,
        'n_test_batches': 10,
        'x_dims': [256, 256, 3],
     }

    def __init__(self, **kwargs):
        self.classification_defaults.update(**kwargs)

        super(Classifier, self).__init__(**self.classification_defaults)
        assert self.sess is not None

        if self.mode=='TRAIN':
            self._training_mode()
        elif self.mode=='TEST':
            self._test_mode()


    def _training_mode(self):
        print 'Setting up {} in training mode'.format(self.name)
        ## ------------------- Input ops ------------------- ##
        self._make_input_ops()

        ## Check for a testing dataset
        if self.dataset.testing_record is not None:
            self.with_test = True

        ## ------------------- Model ops ------------------- ##
        self._make_model_ops()

        ## ------------------- Training ops ------------------- ##
        self.var_list = self._get_update_list()
        self.learning_rate = tf.placeholder_with_default(self.learning_rate, shape=(), name='LR')
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
        self._make_model_ops()

        ## -------------- Gotta have this stuff -------------##
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())


    def _get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if self.name in var.name]

    # def _class_weighted_loss(self):
    #     ## https://github.com/tensorflow/tensorflow/issues/10021
    #     sample_weights = tf.reduce_sum(tf.multiply(self.y_in, self.class_weights), -1)
    #     print '\t segmentation losses sample_weights:', sample_weights
    #     seg_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y_in,
    #         logits=self.y_hat, weights=sample_weights)
    #     print '\t segmentation losses seg_loss:', seg_loss
    #     return seg_loss


    def _make_input_ops(self):
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
            name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.label_op,
            shape=[None, self.n_classes], name='y_in')


    def _make_model_ops(self):
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.training = tf.placeholder_with_default(True, shape=())
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False,
            training=self.training)

        self.y_hat_smax = tf.nn.softmax(self.y_hat)
        print 'Model output y_hat:', self.y_hat.get_shape()


    ## define self.class_loss
    def _make_classification_loss(self, target_op=None):
        self.class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.y_in, logits=target_op)
        self.class_loss = tf.reduce_mean(self.class_loss)


    def _make_training_ops(self):
        with tf.name_scope('classification_losses'):
            self._make_classification_loss(target_op=self.y_hat)

            ## Unused except in pretraining or specificially requested
            self.class_training_op = self.optimizer.minimize(
                self.class_loss, var_list=self.var_list, name='{}_class_train'.format(self.name))

            self.class_loss_sum = tf.summary.scalar('class_loss', self.class_loss)
            self.summary_op_list.append(self.class_loss_sum)

            ## Add in whatever extra losses here
            self.loss = self.class_loss

            self.batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.batch_norm_updates):
                self.train_op = self.optimizer.minimize(self.loss,
                    var_list=self.var_list, name='{}_train'.format(self.name))

            self.class_training_op_list.append(self.train_op)


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
        self.loss_sum = tf.summary.scalar('loss', self.loss)

        ## Scalars:
        self.summary_scalars_op = tf.summary.merge_all()

        ## Images
        # with tf.variable_scope('training_images'):
        #     self.y_in_mask = tf.cast(tf.argmax(self.y_in, axis=-1), tf.float32)
        #     self.y_in_mask = tf.expand_dims(self.y_in_mask, axis=-1)
        #     self.y_hat_mask = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
        #     self.y_hat_mask = tf.cast(self.y_hat_mask, tf.float32)
        #
        #     self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=4)
        #     self.y_in_sum = tf.summary.image('y_in', self.y_in_mask, max_outputs=4)
        #     self.y_hat_sum = tf.summary.image('y_hat', self.y_hat_mask, max_outputs=4)

        ## TODO image summaries for classification.
        # self.summary_images_op = tf.summary.merge(
        #     [self.x_in_sum, self.y_in_sum, self.y_hat_sum])


    def _make_test_ops(self):
        if self.with_test is None:
            print 'WARNING no TEST tfrecord dataset; Skipping test mode'
            return

        with tf.variable_scope('testing_scalars'):
            self.loss_sum_test = tf.summary.scalar('loss_test', self.loss)

        self.summary_test_ops = self.loss_sum_test
        # with tf.variable_scope('testing_images'):
        #     # self.y_in_mask_test = tf.cast(tf.argmax(self.y_in, axis=-1), tf.float32)
        #     # self.y_in_mask_test = tf.expand_dims(self.y_in_mask, axis=-1)
        #     # self.y_hat_mask_test = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
        #     # self.y_hat_mask_test = tf.cast(self.y_hat_mask, tf.float32)
        #
        #     self.x_in_sum_test = tf.summary.image('x_in_test', self.x_in, max_outputs=self.summary_image_n)
        #     self.y_in_sum_test = tf.summary.image('y_in_test', self.y_in_mask, max_outputs=self.summary_image_n)
        #     self.y_hat_sum_test = tf.summary.image('y_hat_test', self.y_hat_mask, max_outputs=self.summary_image_n)
        #
        #     self.summary_test_ops = tf.summary.merge(
        #         [self.loss_sum_test, self.x_in_sum_test,
        #          self.y_in_sum_test, self.y_hat_sum_test])

    def _write_scalar_summaries(self, lr=None):
        if lr is None: lr='constant'
        summary_str, class_loss_ = self.sess.run([self.summary_scalars_op, self.class_loss])
        self.summary_writer.add_summary(summary_str, self.global_step)
        print '[{:07d}] writing scalar summaries (loss={:3.3f}) (lr={:03E})'.format(self.global_step, class_loss_, lr)


    # def _write_image_summaries(self):
    #     print '[{:07d}] writing image summaries'.format(self.global_step)
    #     summary_str = self.sess.run(self.summary_images_op)
    #     self.summary_writer.add_summary(summary_str, self.global_step)


    def inference(self, x_in, keep_prob=1.0):
        feed_dict = {self.x_in: x_in,
                     self.keep_prob: keep_prob,
                     self.training: False}
        y_hat_ = self.sess.run(self.y_hat_smax, feed_dict=feed_dict)
        return y_hat_


    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)


    def test_step(self, step_delta, keep_prob=1.0):
        fd = {self.keep_prob: keep_prob}
        summary_str, test_loss_ = self.sess.run([self.summary_test_ops, self.loss], feed_dict=fd)
        self.summary_writer.add_summary(summary_str, self.global_step+step_delta)
        print '#### BASIC TEST #### [{:07d}] writing test summaries (loss={:3.3f})'.format(self.global_step, test_loss_)


    def train_step(self, keep_prob=1.0):
        self.global_step += 1
        self.sess.run(self.class_training_op_list)

        if self.global_step % self.summary_iters == 0:
            self._write_scalar_summaries()

        # if self.global_step % self.summary_image_iters == 0:
        #     self._write_image_summaries()


    """ Run a number of testing iterations """
    def test(self, keep_prob=1.0):
        ## Switch dataset to testing
        self.dataset._initalize_testing(self.sess)

        for step_delta in xrange(self.n_test_batches):
            self.test_step(step_delta, keep_prob=keep_prob)

        self.dataset._initalize_training(self.sess)
