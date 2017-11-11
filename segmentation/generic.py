import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '.')
from basemodel import BaseModel
from discriminator import ConvDiscriminator

class GenericSegmentation(BaseModel):
    defaults={
        sess: None,
        learning_rate: 1e-3,
        adversarial: False,
        dataset: None,
        x_size: [256, 256],
        conv_kernels: [16, 32],
        deconv_kernels: [32, 16],
        n_classes: None,
        mode: 'TRAIN', }

    def __init__(self, **kwargs):
        defaults.update(kwargs)
        super(GenericSegmentation, self).__init__(**defaults)

        assert self.n_classes is not None
        if self.mode=='TRAIN': assert self.dataset.dstype=='ImageMask'

        ## ------------------- Input ops ------------------- ##
        self.x_in = self.dataset.image_op
        self.y_in = self.dataset.mask_op

        ## ------------------- Model ops ------------------- ##
        self.y_hat = self.model(self.x_in, reuse=False, training=True)
        if self.adversarial:
            self.discriminator = ConvDiscriminator(x_real=self.y_in, x_fake=self.y_hat)
            self.training_op_list += self.discriminator.training_op_list
            self.summary_op_list += self.discriminator.summary_op_list

        self.loss = self.loss_op()
        loss_sum = tf.summary.scalar('loss', self.loss)
        self.summary_op_list.append(loss_sum)

        ## ------------------- Training ops ------------------- ##
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_op = self.optimizer.minimize(self.loss)
        self.training_op_list.append(self.training_op)

        ## ------------------- Gather Summary ops ------------------- ##
        self.summary_op = tf.summary.merge(self.summary_op_list)
        self.training_op_list.append(self.summary_op)

        ## ------------------- TensorFlow helpers ------------------- ##
        self.summary_writer = tf.summary.FileWriter(self.log_dir,
            graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=5)


    def train_step(self):
        summary_str = self.sess.run(self.training_op_list)[0]


    def snapshot(self, step):
        self.saver.save(self.sess, self.save_path, global_step=step)


    def loss_op(self):
        seg_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_in, logits=self.y_hat)
        seg_loss_sum = tf.summary.scalar('seg_loss', seg_loss)
        self.summary_op_list.append(seg_loss_sum)

        if self.adversarial:
            p_real_fake = self.discriminator.p_real_fake
            real_target = tf.zeros_like(p_real_fake)
            adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=real_target, logis=p_real_fake))
            adv_loss_sum = tf.summary.scalar('adv_loss', adv_loss)
            self.summary_op_list.append(adv_loss_sum)
            return seg_loss + adv_loss

        return seg_loss


    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'GenericSegmentation Model'
        with tf.variable_scope('GenericSeg') as scope:
            if reuse:
                scope.reuse_variables()

            c0 = conv(x_in, self.conv_kernels[0], var_scope='c0')
            c0 = lrelu(h0)

            c1 = conv(c0, self.conv_kernels[1], var_scope='c1')
            c1 = lrelu(c1)

            d1 = deconv(c1, self.deconv_kernels[1], var_scope='d1')
            d1 = lrelu(d1)

            d0 = deconv(d1, self.deconv_kernels[0], var_scope='d0')
            d0 = lrelu(d0)

            x_hat = deconv(d0, self.n_classes, var_scope='x_hat')

            return x_hat
