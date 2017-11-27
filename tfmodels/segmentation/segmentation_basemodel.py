import tensorflow as tf
import numpy as np
import sys, os

from ..utilities.basemodel import BaseModel
from discriminator import SegmentationDiscriminator

class SegmentationBaseModel(BaseModel):
    ## Defaults
    segmentation_defaults={
        'adversarial': False,
        'adversary_lr': 1e-4,
        'adversary_lambda': 1,
        'conv_kernels': [32, 64, 128, 256],
        'dataset': None,
        'deconv_kernels': [32, 64],
        'feature_matching': False,
        'learning_rate': 1e-3,
        'log_dir': None,
        'mode': 'TRAIN',
        'name': 'Segmentation',
        'n_classes': None,
        'save_dir': None,
        'sess': None,
        'snapshot_name': 'snapshot',
        'summary_iters': 50,
        'summary_op_list': [],
        'training_op_list': [],
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
                y_fake=tf.nn.softmax(self.y_hat),
                feature_matching=self.feature_matching)
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
        self.snapshot_dir = os.path.join(self.save_dir,
            '{}.ckpt'.format(self.snapshot_name))
        self.saver = tf.train.Saver(max_to_keep=5)
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
        self.y_hat_smax = tf.nn.softmax(self.y_hat)

        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())


    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)


    def make_training_ops(self):
        self.seg_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_in, logits=self.y_hat)
        self.seg_loss = tf.reduce_mean(self.seg_loss)

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

            self.loss = self.seg_loss + self.adversary_lambda * self.adv_loss
        else:
            self.loss = self.seg_loss

        self.training_op = self.optimizer.minimize(
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


    def snapshot(self, step):
        print 'Snapshotting to [{}] step [{}]'.format(self.snapshot_dir, step),
        self.saver.save(self.sess, self.snapshot_dir, global_step=step)
        print 'Done'


    def restore(self, snapshot_path):
        print 'Restoring from {}'.format(snapshot_path)
        try:
            self.saver.restore(self.sess, snapshot_path)
            print 'Success!'
        except:
            print 'Failed! Continuing without loading snapshot.'


    def test_step(self, keep_prob=1.0):
        raise Exception(NotImplementedError)


    def inference(self, x_in, keep_prob=1.0):
        feed_dict = {self.x_in: x_in,
                     self.keep_prob: keep_prob,
                     self.training: False}
        y_hat_ = self.sess.run([self.y_hat_smax], feed_dict=feed_dict)[0]
        return y_hat_


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
