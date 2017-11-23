import tensorflow as tf
import numpy as np

from basemodel import BaseGenerativeModel
from generator import Generator
from discriminator import Discriminator

## defaults for all variables needed in Generator and Discriminator
class GAN(BaseGenerativeModel):
    gan_defaults = {
        'batch_size': 64,
        'dataset': None
        'discriminator': None,
        'dis_learning_rate': 1e-4
        'dis_kernels': [32, 64, 128, 256],
        'generator': None,
        'gen_learning_rate': 2e-4,
        'gen_kernels': [32, 64, 128, 256],
        'sess': None,
        'x_dims': [256, 256, 3],
        'z_dim': 64, }

    def __init__(self, **kwargs):
        self.gan_defaults.update(**kwargs)
        super(GAN, self).__init__(**self.gan_defaults)

        assert self.sess is not None
        assert len(self.x_dims) == 3
        if self.mode=='TRAIN': assert self.dataset is not None

        self.generator = Generator(**self.gan_defaults)
        self.discriminator = Discriminator(**self.gan_defaults)


        ## ---------------------- Input ops ----------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_imds[2]])
        self.z_in_default = tf.random_normal([self.batch_size, self.z_dim],
            mean=0.0, stddev=1.0)
        self.z_in = tf.placeholder_with_default(self.z_in_default,
            shape=[None, self.z_dim])
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')


        ## ---------------------- Model ops ----------------------- ##
        self.x_hat = self.generator.model(self.z_in, keep_prob=self.keep_prob)
        self.p_real_real = self.discriminator.model(self.x_in, keep_prob=self.keep_prob)
        self.p_real_fake = self.discriminator.model(self.x_fake, keep_prob=self.keep_prob, reuse=True)


        ## ---------------------- Loss ops ------------------------ ##
        self._loss_op()

        ## -------------------- Training ops ---------------------- ##
        self._training_ops()

        ## --------------------- Summary ops ---------------------- ##
        self._summary_ops()

        ## ------------------- Done with setup -------------------- ##
        self._print_info_to_file(filename=os.path.join(self.save_dir, 'gan_settings.txt'))



    def _loss_op(self):
        ## Define losses
        self.real_target = tf.ones_like(self.p_real_real)
        self.fake_target = tf.zeros_like(self.p_real_fake)

        ## TODO Label smoothing

        self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.real_target, logits=self.p_real_real ))
        self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.fake_target, logits=self.p_real_fake ))

        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.real_target, logits=self.p_real_fake ))

        self.dis_loss = self.dis_loss_real + self.dis_loss_fake

        self.discriminator_loss_sum = tf.summary.scalar('discriminator_loss', self.dis_loss)
        self.generator_loss_sum = tf.summary.scalar('generator_loss', self.gen_loss)


    def _training_ops(self):
        ## Define training ops
        self.generator_vars = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        self.discriminator_vars = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        self.gen_optimizer = tf.train.AdamOptimizer(self.gen_learning_rate)
        self.dis_optimizer = tf.train.AdamOptimizer(self.dis_learning_rate)

        self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss, var_list=self.generator_vars)
        self.dis_train_op = self.gen_optimizer.minimize(self.dis_loss, var_list=self.discriminator_vars)

        self.training_op_list.append(self.gen_train_op)
        self.training_op_list.append(self.dis_train_op)


    def _summary_ops(self):
        self.x_hat_sum = tf.summary.image('x_hat', self.x_hat, max_outputs=8)
        self.x_in_sum = tf.summary.image('x_in', self.x_in, max_outputs=8)

        self.summary_writer = tf.summary.FileWriter(self.log_dir,
            graph=self.sess.graph, flush_secs=30)

        ## TODO split up scalars and image summaries
        self.summary_op = tf.summary.merge_all()
        self.training_op_list.append(self.summary_op)


    def train_step(self, global_step):
        summary_str = self.sess.run([self.training_op_list])[-1]
        if global_step % self.summary_iters == 0:
            self.summary_writer.add_summary(summary_str, global_step)
