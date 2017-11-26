import tensorflow as tf
import numpy as np

from discriminator_basemodel import BaseDiscriminator
from generator_basemodel import BaseGenerator

''' Generative Adversarial Network

1. Define the disciminator and generator according to the template models

2. implement the loss functions and regularizers

3. ???

'''
class Discriminator(BaseDiscriminator):
    gan_discriminator_defaults = {
        'dis_kernels': [32, 64, 128]
    }

    def __init__(self, **kwargs):
        self.gan_discriminator_defaults.update(**kwargs)
        super(Discriminator, self).__init__(**self.gan_discriminator_defaults)

        self.p_real_fake = self.model(x_in=self.fake)
        self.p_real_real = self.model(x_in=self.real, reuse=True)
        self._make_training_op()

    def model(self, x_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up GAN/Discriminator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            c0 = nonlin(conv(x_in, k_size=7, stride=3, var_scope='c0'))
            c1 = nonlin(conv(c0, self.dis_kernels[0], k_size=7, stride=3, var_scope='c1'))
            c2 = nonlin(conv(c1, self.dis_kernels[1], k_size=7, stride=3, var_scope='c2'))
            flat = tf.contrib.layers.flatten(c2)
            h0 = nonlin(linear(flat, self.dis_kernels[2], var_scope='h0'))
            p_real = linear(h0, 1, var_scope='p_real')

            return p_real



class Generator(BaseGenerator):
    gan_generator_defaults = {
        'gen_kernels': [128, 64, 32],
        'n_upsamples': 3,
        'x_dims': [128, 128, 3],
        # 'z_in': None
    }

    def __init__(self, **kwargs):
        self.gan_generator_defaults.update(**kwargs)
        super(Generator, self).__init__(**self.gan_generator_defaults)

        # self.x_hat = self.model(z_in=self.z_in)

    def model(self, z_in, keep_prob=0.5, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up GAN/Generator'
            print 'Nonlinearity: ', self.nonlin
            nonlin = self.nonlin

            ## Project
            projection = nonlin(linear(self.z_in, self.project_shape, var_scope='projection')) ## [16*16]
            project_conv = tf.reshape(projection, self.resize_shape) ## [16, 16, 1]
            h0 = nonlin(deconv(project_conv, self.gen_kernels[0], var_scope='h0')) ## [32, 32, 128]
            h1 = nonlin(deconv(h0, self.gen_kernels[1], var_scope='h1')) ## [64, 64, 64]
            h2 = nonlin(deconv(h1, self.gen_kernels[2], var_scope='h2')) ## [128, 128, 32]

            x_hat = conv(h2, self.x_dims[-1], stride=1, var_scope='x_hat') ## [128, 128, 3]



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
        'n_upsamples': 3,
        'pretraining': 500,
        'sess': None,
        'soften_labels': False,
        'soften_sddev': 0.01,
        'x_dims': [256, 256, 3],
        'z_dim': 64, }

    def __init__(self, **kwargs):
        self.gan_defaults.update(**kwargs)
        super(GAN, self).__init__(**self.gan_defaults)

        assert self.sess is not None
        assert len(self.x_dims) == 3
        if self.mode=='TRAIN': assert self.dataset is not None

        ## ---------------------- Input ops ----------------------- ##
        self.x_in = tf.placeholder_with_default(self.dataset.image_op,
            shape=[None, self.x_dims[0], self.x_dims[1], self.x_imds[2]])
        self.z_in_default = tf.random_normal([self.batch_size, self.z_dim],
            mean=0.0, stddev=1.0)
        self.z_in = tf.placeholder_with_default(self.z_in_default,
            shape=[None, self.z_dim])
        self.keep_prob = tf.placeholder_with_default(0.5, shape=[], name='keep_prob')

        self.generator = Generator(**self.gan_defaults)
        self.discriminator = Discriminator(**self.gan_defaults)

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

        ## ---------------------- Pretraining --------------------- ##
        self._pretraining()


    def _loss_op(self):
        ## Define losses
        self.real_target = tf.ones_like(self.p_real_real)
        self.fake_target = tf.zeros_like(self.p_real_fake)

        if self.soften_labels:
            real_epsilon = tf.random_normal(shape=tf.shape(real_target),
                mean=0.0, stddev=self.soften_sddev)
            fake_epsilon = tf.random_normal(shape=tf.shape(fake_target),
                mean=0.0, stddev=self.soften_sddev)
            self.real_target = self.real_target + real_epsilon
            self.fake_target = self.fake_target + fake_epsilon

        self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.real_target, logits=self.p_real_real ))
        self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.fake_target, logits=self.p_real_fake ))

        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.real_target, logits=self.p_real_fake ))

        self.discriminator_loss = self.dis_loss_real + self.dis_loss_fake

        self.discriminator_loss_sum = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
        self.generator_loss_sum = tf.summary.scalar('generator_loss', self.generator_loss)


    def _training_ops(self):
        ## Define training ops
        # self.generator_vars = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        # self.discriminator_vars = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        self.generator_vars = self.generator.get_update_list()
        self.discriminator_vars = self.discriminator.get_update_list()

        self.generator_optimizer = tf.train.AdamOptimizer(self.gen_learning_rate)
        self.discriminator_optimizer = tf.train.AdamOptimizer(self.dis_learning_rate)

        self.gen_train_op = self.generator_optimizer.minimize(self.generator_loss, var_list=self.generator_vars)
        self.dis_train_op = self.discriminator_optimizer.minimize(self.discriminator_loss, var_list=self.discriminator_vars)

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


    def _pretraining(self):
        print 'Pretraining discriminator'
        for _ in xrange(self.pretraining):
            _ = self.sess.run([self.dis_train_op])

        print 'Pretraining generator'
        for _ in xrange(self.pretraining):
            _ = self.sess.run([self.gen_train_op])

        summary_str = self.sess.run(self.training_op_list)[-1]
        self.summary_writer.add_summary(summary_str, 0)


    def train_step(self, global_step):
        summary_str = self.sess.run([self.training_op_list])[-1]
        if global_step % self.summary_iters == 0:
            self.summary_writer.add_summary(summary_str, global_step)
