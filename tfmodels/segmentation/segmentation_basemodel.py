import tensorflow as tf
import numpy as np
import sys, os

from ..utilities.basemodel import BaseModel
# from ..utilities.ops import class_weighted_pixelwise_crossentropy
from discriminator import SegmentationDiscriminator

class SegmentationBaseModel(BaseModel):
    ## Defaults
    segmentation_defaults={
        'adversarial': False,
        'adversary_lr': 1e-4,
        'adversary_lambda': 1,
        'adversary_feature_matching': False,
        'class_weights': None,
        'conv_kernels': [32, 64, 128, 256],
        'dataset': None,
        'deconv_kernels': [32, 64],
        'global_step': 0,
        'k_size': 3,
        'learning_rate': 1e-3,
        'log_dir': None,
        'mode': 'TRAIN',
        'name': 'Segmentation',
        'n_classes': None,
        'pretraining': False,
        'pretrain_g': 2500,
        'pretrain_d': 1000,
        'save_dir': None,
        'sess': None,
        'seg_training_op_list': [],
        'snapshot_name': 'snapshot',
        'summary_iters': 50,
        'summary_op_list': [],
        'x_dims': [256, 256, 3],
     }


    def __init__(self, **kwargs):
        self.segmentation_defaults.update(**kwargs)

        super(SegmentationBaseModel, self).__init__(**self.segmentation_defaults)
        assert self.sess is not None

        ## Set the nonlinearity for all downstream models
        self.nonlin = tf.nn.selu

        if self.mode=='TRAIN':
            self._training_mode()
        elif self.mode=='TEST':
            self._test_mode()


    def _training_mode(self):
        print 'Setting up {} in test mode'.format(self.name)
        ## ------------------- Input ops ------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_dims[2]],
                name='x_in')
        self.y_in = tf.placeholder_with_default(self.dataset.mask_op,
            shape=[None, self.x_dims[0], self.x_dims[1], 1], name='y_in')
        if self.y_in.get_shape().as_list()[-1] != self.n_classes:
            self.y_in_mask = tf.cast(tf.identity(self.y_in), tf.float32)
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
        print 'Model output y_hat:', self.y_hat.get_shape()

        ## ------------------- Training ops ------------------- ##
        self.var_list = self.get_update_list()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
            name='{}_Adam'.format(self.name))

        if self.adversarial:
            self.discriminator = SegmentationDiscriminator(sess=self.sess,
                x_in=self.x_in,
                y_real=self.y_in,
                # y_fake=tf.nn.softmax(self.y_hat),
                y_fake=tf.nn.sigmoid(self.y_hat),
                learning_rate=self.adversary_lr,
                feature_matching=self.adversary_feature_matching)
            self.discriminator.print_info()
            self.seg_training_op_list.append(self.discriminator.train_op)

        self.make_training_ops()

        ## ------------------- Gather Summary ops ------------------- ##
        self.summary_op_list += self.summaries()
        # self.summary_op = tf.summary.merge(self.summary_op_list)
        self.summary_op = tf.summary.merge_all()
        # self.training_op_list.append(self.summary_op)

        ## ------------------- TensorFlow helpers ------------------- ##
        self.summary_writer = tf.summary.FileWriter(self.log_dir,
            graph=self.sess.graph, flush_secs=30)
        ## Append a model name to the save path
        self.snapshot_path = os.path.join(self.save_dir,
            '{}.ckpt'.format(self.snapshot_name))
        # self.make_saver() ## In progress (SAVE1)
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

        self._print_info_to_file(filename=os.path.join(self.save_dir,
            '{}_settings.txt'.format(self.name)))

        if self.adversarial:
            self.discriminator._print_info_to_file(filename=os.path.join(self.save_dir,
                '{}_settings.txt'.format(self.discriminator.name)))

        ## Somehow calling this during init() makes it not work
        # if self.adversarial and self.pretraining_iters:
        #     self.pretrain()

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
        self.y_hat_smax = tf.nn.softmax(self.y_hat)

        # self.make_saver() ## In progress (SAVE1)
        self.saver = tf.train.Saver(max_to_keep=5)

        self.sess.run(tf.global_variables_initializer())


    def bayesian_inference(self, x_in, samples=25, keep_prob=0.5, ret_all=False):
        assert keep_prob < 1.0
        assert samples > 1
        assert x_in.shape[0] == 1 and len(x_in.shape) == 4

        x_in_stack = np.concatenate([x_in]*samples, axis=0)

        y_hat = self.inference(x_in=x_in_stack, keep_prob=keep_prob)
        y_bar_mean = np.mean(y_hat, axis=0) ## (1, h, w, n_classes)

        if ret_all:
            y_bar_var = np.var(y_hat, axis=0)
            y_bar_argmax = np.argmax(y_bar_mean, axis=-1)
            y_bar_argmax = np.expand_dims(y_bar_argmax, 0)
            return y_bar_mean, y_bar_var, y_bar_argmax
        else:
            return y_bar_mean


    def get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if self.name in var.name]


    def inference(self, x_in, keep_prob=1.0):
        feed_dict = {self.x_in: x_in,
                     self.keep_prob: keep_prob,
                     self.training: False}
        y_hat_ = self.sess.run([self.y_hat_smax], feed_dict=feed_dict)[0]
        return y_hat_


    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)


    def make_training_ops(self):
        with tf.name_scope('segmentation_losses'):
            if self.class_weights:
                ## https://github.com/tensorflow/tensorflow/issues/10021
                sample_weights = tf.reduce_sum(tf.multiply(self.y_in, self.class_weights), -1)
                print '\t segmentation losses sample_weights:', sample_weights
                self.seg_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y_in,
                    logits=self.y_hat, weights=sample_weights)
                print '\t segmentation losses seg_loss:', self.seg_loss

                # self.seg_loss = class_weighted_pixelwise_crossentropy(
                #     labels=self.y_in, logits=self.y_hat, weights=self.class_weights)
                # self.seg_loss = tf.reduce_mean(self.seg_loss)
                # print '\t segmentation losses seg_loss:', self.seg_loss
            else:
                self.seg_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y_in, logits=self.y_hat)
                self.seg_loss = tf.reduce_mean(self.seg_loss)

            ## Unused except in pretraining or specificially requested
            self.seg_training_op = self.optimizer.minimize(
                self.seg_loss, var_list=self.var_list, name='{}_seg_train'.format(self.name))

            self.seg_loss_sum = tf.summary.scalar('seg_loss', self.seg_loss)
            self.summary_op_list.append(self.seg_loss_sum)

            if self.adversarial:
                ## Train the generator w.r.t. the current discriminator
                ## The discriminator itself is updated elsewhere
                p_real_fake = self.discriminator.p_real_fake
                real_target = tf.ones_like(p_real_fake)
                self.adv_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=real_target, logits=p_real_fake)
                self.adv_loss = tf.reduce_mean(self.adv_loss)

                self.adv_loss_sum = tf.summary.scalar('adv_loss', self.adv_loss)
                self.summary_op_list.append(self.adv_loss_sum)

                ## Combined segmentation and adversarial terms
                self.loss = self.seg_loss + \
                    self.adversary_lambda * self.adv_loss

                ## Alternative feature matching term
                if self.adversary_feature_matching:
                    print 'Setting up adversarial feature matching'
                    features_fake = self.discriminator.fake_features
                    features_real = self.discriminator.real_features

                    self.adv_feature_loss = tf.losses.mean_squared_error(
                        labels=features_real, predictions=features_fake )

                    self.adv_feature_loss_sum = tf.summary.scalar(
                        'adv_feature_loss', self.adv_feature_loss)
                    self.summary_op_list.append(self.adv_feature_loss_sum)

                    ## Combined segmentation and adversarial terms
                    self.loss = self.seg_loss + \
                        self.adversary_lambda * self.adv_feature_loss + \
                        self.adversary_lambda * self.adv_loss

                    ## Separate ops for the two losses
                    # self.adv_feature_training_op = self.optimizer.minimize(
                    #     self.adv_feature_loss, var_list=self.var_list)
                    # self.training_op_list.append(self.adv_feature_training_op)
                    # self.loss = self.seg_loss

                # self.training_op_list.append(self.discriminator.train_op)
            else:
                self.loss = self.seg_loss

            self.train_op = self.optimizer.minimize(
                self.loss, var_list=self.var_list, name='{}_train'.format(self.name))

            self.seg_training_op_list.append(self.train_op)


    def pretrain(self):
        print 'Pretraining Generator without adversary for {} iterations'.format(self.pretrain_g)
        for _ in xrange(self.pretrain_g):
            self.global_step += 1
            self.sess.run([self.seg_training_op])
            # print '\titeration {}'.format(self.global_step)
            if self.global_step % self.summary_iters == 0:
                # print 'Saving summary'
                # summary_str = self.sess.run([self.summary_op_list])[-1]
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.global_step)


        print 'Pretraining Discriminator for {} iterations'.format(self.pretrain_d)
        for _ in xrange(self.pretrain_d):
            self.global_step += 1
            self.sess.run(self.discriminator.discriminator_train_op_list)
            if self.global_step % self.summary_iters == 0:
                # print 'Saving summary'
                # summary_str = self.sess.run([self.summary_op_list])[-1]
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.global_step)



    def restore(self, snapshot_path):
        print 'Restoring from {}'.format(snapshot_path)
        try:
            self.saver.restore(self.sess, snapshot_path)
            print 'Success!'

            ## In progress for restoring model + discriminator separately (SAVE1)
            # for saver, snap_path in zip(self.saver):
        except:
            print 'Failed! Continuing without loading snapshot.'


    def snapshot(self):
        ## In progress for saving model + discriminator separately (SAVE1)
        ## have to work up the if/else logic upstream first
        # for saver, snap_dir in zip(self.saver_list, self.snap_dir_list):
        #     print 'Snapshotting to [{}] step [{}]'.format(snap_dir, step),
        #     saver.save(self.sess, snap_dir, global_step=step)

        print 'Snapshotting to [{}] step [{}]'.format(self.snapshot_path, self.global_step),
        self.saver.save(self.sess, self.snapshot_path, global_step=self.global_step)

        print 'Done'


    def summaries(self):
        ## Input image
        self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=4)
        self.y_in_sum = tf.summary.image('y_in', self.y_in_mask, max_outputs=4)
        self.y_hat_sum = tf.summary.image('y_hat', self.y_hat_mask, max_outputs=4)
        ## Loss scalar
        self.loss_sum = tf.summary.scalar('loss', self.loss)
        ## TODO Filters

        return [
            self.x_in_sum,
            self.y_in_sum,
            self.y_hat_sum,
            self.loss_sum]


    ## TODO -- maybe
    def test_step(self, keep_prob=1.0):
        raise Exception(NotImplementedError)


    def train_step(self):
        self.global_step += 1
        self.sess.run(self.seg_training_op_list)
        if self.global_step % self.summary_iters == 0:
            summary_str = self.sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.global_step)
