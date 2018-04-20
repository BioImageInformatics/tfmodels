from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys, os

from segmentation_basemodel import Segmentation

class SegmentationBayesian(Segmentation):
    ## Defaults. arXiv links correspond to inspirational materials
    bayesian_segmentation_defaults={
        'class_weights': None, ## https://arxiv.org/abs/1511.00561
        'dataset': None,
        'aleatoric': False, ## https://arxiv.org/abs/1703.04977
        'aleatoric_T': 25,
        'epistemic_T': 10,
        'global_step': 0,
        'k_size': 3,
        'learning_rate': 1e-3,
        'log_dir': None,
        'mode': 'TRAIN',
        'name': 'Segmentation',
        'n_classes': None,
        'save_dir': None,
        'sess': None,
        'seg_training_op_list': [],
        'summarize_grads': False,
        'summary_iters': 50,
        'summary_image_iters': 250,
        'summary_op_list': [],
        'x_dims': [256, 256, 3],
     }

    def __init__(self, **kwargs):
        self.bayesian_segmentation_defaults.update(**kwargs)
        super(SegmentationBayesian, self).__init__(**self.bayesian_segmentation_defaults)


    def _make_input_ops(self):
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
            name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.mask_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.n_classes], name='y_in')


    def _make_model_ops(self, keep_prob=0.5, training=True):
        self.keep_prob = tf.placeholder_with_default(keep_prob, shape=[], name='keep_prob')
        self.training = tf.placeholder_with_default(training, shape=())
        self.y_hat, self.sigma = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False,
            training=self.training)

        self.y_hat_smax = tf.nn.softmax(self.y_hat, axis=-1)
        print('Model output y_hat:', self.y_hat.get_shape())
        print('Model output sigma:', self.sigma.get_shape())


    def _heteroscedastic_aleatoric_loss(self):
        assert self.sigma is not None, 'Model does not have attribute sigma. Define sigma in model()'

        print('Setting up heteroscedastic aleatoric loss:')
        ## Make a summary for sigma
        self.sigma_summary = tf.summary.scalar('sigma_mean', tf.reduce_mean(self.sigma))
        self.summary_op_list.append(self.sigma_summary)

        with tf.variable_scope('aleatoric') as scope:
            ## (batch_size, h*w, n_classes)
            sigma_v = tf.reshape(self.sigma, [-1, np.prod(self.x_dims[:2]), 1])
            dist = tf.distributions.Normal(loc=0.0, scale=sigma_v, name='dist')
            y_hat_v = tf.reshape(self.y_hat, [-1, np.prod(self.x_dims[:2]), self.n_classes])
            y_in_v = tf.reshape(self.y_in, [-1, np.prod(self.x_dims[:2]), self.n_classes])

            print('\t sigma_v', sigma_v.get_shape())
            print('\t y_hat_v', y_hat_v.get_shape())
            print('\t y_in_v', y_in_v.get_shape())

            def _corrupt_with_noise(yhat, dist):
                ## sample is the same shape as the sigma output
                epsilon = tf.transpose(dist.sample(self.n_classes), perm=[1,2,0,3]) ## LOL
                epsilon = tf.squeeze(epsilon)
                return yhat + epsilon

            def loss_fn(yhat):
                y_hat_eps = _corrupt_with_noise(yhat, dist)
                print('\t y_hat_eps', y_hat_eps.get_shape())

                if self.class_weights:
                    sample_weights = tf.reduce_sum(tf.multiply(y_in_v, self.class_weights), -1)
                    print('\t segmentation losses sample_weights:', sample_weights)
                    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_in_v,
                        logits=y_hat_eps, weights=sample_weights)
                    print('\t segmentation losses loss:', loss)
                    return loss
                else:
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_in_v, logits=y_hat_eps)
                    print('\t loss', loss.get_shape())
                    return loss

            y_hat_tile = tf.tile(tf.expand_dims(y_hat_v, 0), [self.aleatoric_T,1,1,1], name='y_hat_tile')
            print('y_hat_tile', y_hat_tile.get_shape())

            loss_fn_map = lambda x: loss_fn(x)
            losses = tf.map_fn(loss_fn_map, y_hat_tile)
            print('losses', losses.get_shape())

            ## Sum over classes
            # loss = tf.reduce_sum(losses, axis=-1)
            # print 'loss', loss.get_shape()

            ## Average over pixels
            # loss = tf.reduce_mean(loss, axis=-1)

            ## Average over batch, pixels and T
            self.seg_loss = tf.reduce_mean(losses)


    def _make_training_ops(self):
        print('Bayesian model setting up losses')
        with tf.name_scope('segmentation_losses'):
            ## Logic to define the correct version of segmentation loss
            ## BUG if aleatoric is requested, the constructor will ignore
            ##  the request for weighted classes
            self._heteroscedastic_aleatoric_loss()

            ## Unused except in pretraining or specificially requested
            self.seg_training_op = self.optimizer.minimize(
                self.seg_loss, var_list=self.var_list, name='{}_seg_train'.format(self.name))

            self.seg_loss_sum = tf.summary.scalar('seg_loss', self.seg_loss)
            self.summary_op_list.append(self.seg_loss_sum)

            self.loss = self.seg_loss

            self.train_op = self.optimizer.minimize(
                self.loss, var_list=self.var_list, name='{}_train'.format(self.name))

            self.seg_training_op_list.append(self.train_op)


    def _make_summaries(self):
        ## https://github.com/aymericdamien/ \
        ## TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
        if self.summarize_grads:
            self.summary_gradient_list = []
            grads = tf.gradients(self.loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            for grad, var in grads:
                try:
                    self.summary_gradient_list.append(
                        tf.summary.histogram(var.name + '/gradient', grad))
                except:
                    print('Failed to make summary for: {}'.format(var.name))

        ## Loss scalar
        self.loss_sum = tf.summary.scalar('loss', self.loss)

        ## Scalars:
        self.summary_scalars_op = tf.summary.merge_all()

        ## Images
        with tf.variable_scope('training_images'):
            self.y_in_mask = tf.cast(tf.argmax(self.y_in, axis=-1), tf.float32)
            self.y_in_mask = tf.expand_dims(self.y_in_mask, axis=-1)
            self.y_hat_mask = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
            self.y_hat_mask = tf.cast(self.y_hat_mask, tf.float32)

            self.sigma_sum = tf.summary.image('sigma_img', self.sigma, max_outputs=self.summary_image_n)

            self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=self.summary_image_n)
            self.y_in_sum = tf.summary.image('y_in', self.y_in_mask, max_outputs=self.summary_image_n)
            self.y_hat_sum = tf.summary.image('y_hat', self.y_hat_mask, max_outputs=self.summary_image_n)

        ## TODO Filters
        self.summary_images_op = tf.summary.merge(
            [self.x_in_sum, self.y_in_sum, self.y_hat_sum, self.sigma_sum])


    def _make_test_ops(self):
        if self.with_test is None:
            print('WARNING no TEST tfrecord dataset; Skipping test mode')
            return

        with tf.variable_scope('testing_scalars'):
            self.loss_sum_test = tf.summary.scalar('loss_test', self.loss)

        with tf.variable_scope('testing_images'):
            # self.y_in_mask_test = tf.cast(tf.argmax(self.y_in, axis=-1), tf.float32)
            # self.y_in_mask_test = tf.expand_dims(self.y_in_mask, axis=-1)
            # self.y_hat_mask_test = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
            # self.y_hat_mask_test = tf.cast(self.y_hat_mask, tf.float32)

            self.sigma_sum_test = tf.summary.image('sigma_img_test', self.sigma, max_outputs=self.summary_image_n)

            self.x_in_sum_test = tf.summary.image('x_in_test', self.x_in, max_outputs=self.summary_image_n)
            self.y_in_sum_test = tf.summary.image('y_in_test', self.y_in_mask, max_outputs=self.summary_image_n)
            self.y_hat_sum_test = tf.summary.image('y_hat_test', self.y_hat_mask, max_outputs=self.summary_image_n)

            self.summary_test_ops = tf.summary.merge(
                [self.loss_sum_test, self.x_in_sum_test,
                 self.y_in_sum_test, self.y_hat_sum_test,
                 self.sigma_sum_test])


    """
    Inference function needs to perturb output based on sigma
    """
    def inference(self, x_in, keep_prob=1.0):
        feed_dict = {self.x_in: x_in,
                     self.keep_prob: keep_prob,
                     self.training: False}
        y_hat_, sigma_ = self.sess.run([self.y_hat, self.sigma], feed_dict=feed_dict)
        return y_hat_, sigma_


    """ function for approximate bayesian inference via dropout
    if ret_all is true, then we return the mean, variance and sigma images
    if ret_all is false, then we return just the softmaxed yhat
    """
    def bayesian_inference(self, x_in, samples=25, keep_prob=0.5, ret_all=False):
        assert keep_prob < 1.0
        assert x_in.shape[0] == 1 and len(x_in.shape) == 4

        ## Option A: (one big iteration)
        # x_in_stack = np.concatenate([x_in]*samples, axis=0)
        # y_hat = self.inference(x_in=x_in_stack, keep_prob=keep_prob)
        # y_bar_mean = np.mean(y_hat, axis=0) ## (1, h, w, n_classes)

        ## Option B: (smaller iterations)
        y_hat_sigmas = [self.inference(x_in=x_in, keep_prob=keep_prob) for _ in xrange(samples)]
        y_hat = [x[0] for x in y_hat_sigmas]
        y_hat = np.stack(y_hat, axis=0)
        y_bar_mean = np.mean(y_hat, axis=0)

        sigmas = [x[1] for x in y_hat_sigmas]
        sigmas = np.stack(sigmas, axis=0)

        if ret_all:
            y_bar_var = np.var(y_hat, axis=0)
            sigma_bar = np.mean(sigmas, axis=0)
            # y_bar_argmax = np.argmax(y_bar_mean, axis=-1)
            # y_bar_argmax = np.expand_dims(y_bar_argmax, 0)
            return y_bar_mean, y_bar_var, sigma_bar
        else:
            return y_bar_mean
