from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys, os

from .segmentation_basemodel import Segmentation
from .discriminator import SegmentationDiscriminator

class SegmentationAdversarial(Segmentation):

    def __init__(self, **kwargs):
        ## Defaults. arXiv links correspond to inspirational materials
        adversarial_segmentation_defaults={
            'adversary': False, ## https://arxiv.org/abs/1611.08408
            'adversary_lr': 1e-4,
            'adversary_lambda': 1,
            'adversary_feature_matching': False, ## https://arxiv.org/abs/1606.03498
            'class_weights': None, ## https://arxiv.org/abs/1511.00561
            'dataset': None,
            'global_step': 0,
            'k_size': 3,
            'learning_rate': 1e-3,
            'log_dir': None,
            'mode': 'TRAIN',
            'name': 'SegmentationAdversarial',
            'n_classes': None,
            'pretrain_g': 2500,
            'pretrain_d': 1000,
            'save_dir': None,
            'sess': None,
            'seg_training_op_list': [],
            'summarize_grads': False,
            'summary_iters': 50,
            'summary_image_iters': 250,
            'summary_op_list': [],
            'x_dims': [256, 256, 3],
         }
        adversarial_segmentation_defaults.update(**kwargs)

        super(SegmentationAdversarial, self).__init__(**adversarial_segmentation_defaults)
        assert self.sess is not None

        ## Set the nonlinearity for all downstream models
        self.nonlin = tf.nn.selu

        if self.mode=='TRAIN':
            self._training_mode()
        elif self.mode=='TEST':
            self._test_mode()

    def _training_mode(self):
        print('Setting up {} in training mode'.format(self.name))
        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.mask_op,
            shape=[None, self.x_dims[0], self.x_dims[1], 1], name='y_in')

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False)

        self.y_hat_smax = tf.nn.softmax(self.y_hat)
        self.y_hat_mask = tf.expand_dims(tf.argmax(self.y_hat, -1), -1)
        self.y_hat_mask = tf.cast(self.y_hat_mask, tf.float32)
        print('Model output y_hat:', self.y_hat.get_shape())

        ## ------------------- Training ops ------------------- ##
        self.var_list = self.get_update_list()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
            name='{}_Adam'.format(self.name))

        if self.adversary:
            self.discriminator = SegmentationDiscriminator(sess=self.sess,
                x_in=self.x_in,
                y_real=self.y_in,
                # y_fake=tf.nn.softmax(self.y_hat),
                y_fake=tf.nn.sigmoid(self.y_hat),
                learning_rate=self.adversary_lr,
                feature_matching=self.adversary_feature_matching)
            self.discriminator.print_info()
            self.seg_training_op_list.append(self.discriminator.train_op)

        self._make_training_ops()

        ## ------------------- Gather Summary ops ------------------- ##
        self._make_summaries()

        ## ------------------- TensorFlow helpers ------------------- ##
        self._tf_ops()
        self.sess.run(tf.global_variables_initializer())

        self._print_info_to_file(filename=os.path.join(self.save_dir,
            '{}_settings.txt'.format(self.name)))

        if self.adversary:
            self.discriminator._print_info_to_file(filename=os.path.join(self.save_dir,
                '{}_settings.txt'.format(self.discriminator.name)))

        ## Somehow calling this during init() makes it not work
        # if self.adversary and self.pretraining_iters:
        #     self.pretrain()

    def _test_mode(self):
        print('Setting up {} in inference mode'.format(self.name))
        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder('float',
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
            name='x_in')

        ## ------------------- Model ops ------------------- ##
        # self.keep_prob = tf.placeholder('float', name='keep_prob')
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')
        self.y_hat = self.model(self.x_in, keep_prob=self.keep_prob, reuse=False)
        self.y_hat_smax = tf.nn.softmax(self.y_hat)

        # self.make_saver() ## In progress (SAVE1)
        # with tf.device('/cpu:0'):
        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess.run(tf.global_variables_initializer())


    ## define self.loss
    def _adversarial_feature_matching_loss(self):
        ## Feature matching style -- I have it set to always to _adversarial_loss()
        ## and then add this to the end of it
        print('Setting up adversarial feature matching')
        features_fake = self.discriminator.fake_features
        features_real = self.discriminator.real_features

        self.adv_feature_loss = tf.losses.mean_squared_error(
            labels=features_real, predictions=features_fake )

        self.adv_feature_loss_sum = tf.summary.scalar(
            'adv_feature_loss', self.adv_feature_loss)
        self.summary_op_list.append(self.adv_feature_loss_sum)

        ## Combined segmentation and adversarial terms
        ## Replace the old self.loss
        self.loss = self.seg_loss + \
            self.adversary_lambda * self.adv_feature_loss + \
            self.adversary_lambda * self.adv_loss



    ## define self.loss
    def _adversarial_loss(self):
        ## Straight up maximize p(real | g(z))
        p_real_fake = self.discriminator.p_real_fake
        real_target = tf.ones_like(p_real_fake)
        self.adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=real_target, logits=p_real_fake)
        self.adv_loss = tf.reduce_mean(self.adv_loss)

        self.adv_loss_sum = tf.summary.scalar('adv_loss', self.adv_loss)
        self.summary_op_list.append(self.adv_loss_sum)

        self.loss = self.seg_loss + \
            self.adversary_lambda * self.adv_loss


    def _make_training_ops(self):
        with tf.name_scope('segmentation_losses'):
            ## Unused except in pretraining or specificially requested
            self.seg_training_op = self.optimizer.minimize(
                self.seg_loss, var_list=self.var_list, name='{}_seg_train'.format(self.name))

            self.seg_loss_sum = tf.summary.scalar('seg_loss', self.seg_loss)
            self.summary_op_list.append(self.seg_loss_sum)

            ## Make self.loss operation with combinations of different losses
            if self.adversary:
                ## Train the generator w.r.t. the current discriminator
                ## The discriminator itself is updated elsewhere
                self._adversarial_loss()

                ## Alternative feature matching term
                if self.adversary_feature_matching:
                    self._adversarial_feature_matching_loss()
            else:
                ## No special
                self.loss = self.seg_loss

            self.train_op = self.optimizer.minimize(
                self.loss, var_list=self.var_list, name='{}_train'.format(self.name))

            self.seg_training_op_list.append(self.train_op)


    # def _make_summaries(self):
    #     ## https://github.com/aymericdamien/ \
    #     ## TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py
    #     if self.summarize_grads:
    #         self.summary_gradient_list = []
    #         grads = tf.gradients(self.loss, tf.trainable_variables())
    #         grads = list(zip(grads, tf.trainable_variables()))
    #         for grad, var in grads:
    #             self.summary_gradient_list.append(
    #                 tf.summary.histogram(var.name + '/gradient', grad))
    #
    #     ## Loss scalar
    #     self.loss_sum = tf.summary.scalar('loss', self.loss)
    #
    #     ## Scalars:
    #     ## Merge all before crating the image summary ops
    #     self.summary_scalars_op = tf.summary.merge_all()
    #
    #     ## Images
    #     self.y_in_mask = tf.cast(tf.argmax(self.y_in, axis=-1), tf.float32)
    #     self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=4)
    #     self.y_in_sum = tf.summary.image('y_in', self.y_in_mask, max_outputs=4)
    #     self.y_hat_sum = tf.summary.image('y_hat', self.y_hat_mask, max_outputs=4)
    #
    #     ## TODO Filters
    #
    #     self.summary_images_op = tf.summary.merge(
    #         [self.x_in_sum, self.y_in_sum, self.y_hat_sum])


    def pretrain(self):
        if not self.adversary:
            print('Pretraining requested but this model is not in adversarial mode')
            print('use adversary=True to include adversarial training')
            print('Continuing')
            return

        print('Pretraining Generator without adversary for {} iterations'.format(self.pretrain_g))
        for _ in xrange(self.pretrain_g):
            self.global_step += 1
            self.sess.run([self.seg_training_op])
            if self.global_step % self.summary_iters == 0:
                self._write_scalar_summaries()
            if self.global_step % self.summary_image_iters == 0:
                self._write_scalar_summaries()

        print('Pretraining Discriminator for {} iterations'.format(self.pretrain_d))
        for _ in xrange(self.pretrain_d):
            self.global_step += 1
            self.sess.run(self.discriminator.discriminator_train_op_list)
            if self.global_step % self.summary_iters == 0:
                self._write_scalar_summaries()
            if self.global_step % self.summary_image_iters == 0:
                self._write_scalar_summaries()
