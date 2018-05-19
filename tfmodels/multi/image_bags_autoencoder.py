from __future__ import print_function
import tensorflow as tf
import sys

from ..utilities.ops import *
from ..generative.encoder_basemodel import BaseEncoder
from ..generative.generator_basemodel import BaseGenerator
from ..utilities.basemodel import BaseModel

"""
DeepSets / Multi-instance / repeated measures

we want to tell if the positive class is present in a bag
the training mini batches are like:

x = [batch_size, samples, dimensions]
y = [batch_size]

we want to predict which batches contain at least one positive observation

models are structured to take in [samples, dimensions] input tensors,
or [1, dimensions] inputs

we process x --> z with a learnable, nonlinear function

in this setting, x --> z amounts to learning a binary classifier for "is positive"
for each x.
"""

class Encoder(BaseEncoder):

    def __init__(self, name='convolutional_encoder', **kwargs):
        encoder_defaults = {
            'kernels': [64, 128],
            'hidden_dim': [512],
            'name': 'Encoder',
            }
        encoder_defaults.update(**kwargs)
        super(Encoder, self).__init__(**encoder_defaults)

        self.name = name

    def model(self, x_in, keep_prob=0.5, reuse=False, return_mode='z_bar_x'):
        print('Setting up Encoder model')
        print('\t x_in: ', x_in.get_shape())
        with tf.variable_scope(self.name) as scope:
            if reuse:
                print('\t Reusing variables')
                scope.reuse_variables()

            nonlin = tf.nn.selu

            c0 = nonlin(conv(x_in, self.kernels[0], k_size=5, stride=2, var_scope='c0'))
            c0_p = tf.nn.max_pool(c0, [1,2,2,1], [1,2,2,1], padding='VALID')
            c1 = nonlin(conv(c0_p, self.kernels[1], k_size=3, stride=2, var_scope='c1'))
            c1_flat = tf.contrib.layers.flatten(c1)

            h0 = nonlin(linear(c1_flat, self.hidden_dim[0], var_scope='h0'))

            z_i = linear(h0, self.z_dim, var_scope='z_i')

            ## Predict y from z -- or split network heads
            y_i0 = nonlin(linear(z_i, self.z_dim, var_scope='y_i0'))
            # y_i0 = nonlin(linear(h0, self.z_dim, var_scope='y_i0'))
            y_i = linear(y_i0, self.n_classes, var_scope='y_i')

            ## Reduce mean
            z_bar_x = tf.reduce_mean(z_i, axis=0)
            y_bar_x = tf.reduce_mean(y_i, axis=0)

            print('\t Encoder heads:')
            print('\t z_i', z_i.get_shape())
            print('\t y_i', y_i.get_shape())
            print('\t z_bar_x', z_bar_x.get_shape())
            print('\t y_bar_x', y_bar_x.get_shape())

            ## super weird
            if return_mode == 'z_bar_x':
                return z_bar_x
            elif return_mode == 'y_bar_x':
                return y_bar_x
            elif return_mode == 'z_i':
                return z_i
            elif return_mode == 'y_i':
                return y_i



class Generator(BaseGenerator):
    deflt = {
        'gen_kernels': [128, 64],
        'name': 'Generator',
        'x_dims': [28, 28, 1],
    }

    def __init__(self, **kwargs):
        self.deflt.update(**kwargs)
        super(Generator, self).__init__(**self.deflt)

    def model(self, z_in, reuse=False):
        print('Setting up Generator model')
        print('\t z_in:', z_in.get_shape())
        with tf.variable_scope(self.name) as scope:
            if reuse:
                print('Reusing variables')
                scope.reuse_variables()

            nonlin = tf.nn.selu

            projection = nonlin(linear(z_in, self.project_shape, var_scope='projection'))
            project_conv = tf.reshape(projection, self.resize_shape)
            h0 = nonlin(deconv(project_conv, self.gen_kernels[0], k_size=4, var_scope='h0'))
            h1 = nonlin(deconv(h0, self.gen_kernels[1], k_size=4, var_scope='h1'))

            x_hat_logit = conv(h1, self.x_dims[-1], k_size=3, stride=1, var_scope='x_hat')
            # x_hat = tf.nn.sigmoid(x_hat_logit)

            return x_hat_logit


"""
Extends the bagged model by adding an autoencoder - reconstruction loss
Also, force z to be normal gaussian
"""
class ImageBagAutoencoder(BaseModel):
    image_bag_default = {
        'dataset': None,
        'learning_rate': 1e-4,
        'n_classes': 2,
        'name': 'deepsets',
        'no_classifier': True,
        'sess': None,
        'summarize_grads': False,
        'summarize_vars': False,
        'summary_iter': 10,
        'x_dim': [28, 28, 1],
        'z_dim': 32
        }

    def __init__(self, name='deepsets', **kwargs):
        self.image_bag_default.update(**kwargs)
        super(ImageBagAutoencoder, self).__init__(**self.image_bag_default)

        self.name = name
        assert self.sess is not None

        ## first two dimensions of x_in should be None
        assert len(self.x_dim) == 3
        self.x_in = tf.placeholder('float', shape=[None, None]+self.x_dim)
        self.y_in = tf.placeholder('float', shape=[None, self.n_classes], name='y_in')

        self.x_individual = tf.placeholder('float', shape=[None]+self.x_dim, name='x_individual')
        self.z_individual = tf.placeholder('float', shape=[None, self.z_dim], name='z_individual')

        ## use n_classes to get a discriminiator trained to detect positive x_i

        self.encoder = Encoder(z_dim=self.z_dim, n_classes=self.n_classes)
        self.encoder.print_info()

        self.generator = Generator()
        self.generator.print_info()

        self.model(self.x_in)

        ## ------------------- Loss ops
        self._loss_op()

        ## ------------------- Training ops
        self._training_ops()

        ## ------------------- Summary ops
        self._summary_ops()

        ## ------------------- TF housekeeping
        self._tf_ops()
        self.sess.run(tf.global_variables_initializer())


    ## still want model to return, but save hooks to intermediate ops as class attr.
    def model(self, x_in, keep_prob=0.5, reuse=False):
        print('\t Setting up Classification model')
        print('\t Using name scope: {}'.format(self.name))
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            nonlin = self.nonlin

            ## Initialize the encoder
            self.y_individual = self.encoder.model(self.x_individual, return_mode='y_i')
            print('\t y_individual', self.y_individual.get_shape())

            ## Get average y for each sample set
            self.y_bar_map_fn = lambda x: self.encoder.model(x, return_mode='y_bar_x', reuse=True)
            self.y_bar_x = tf.map_fn(self.y_bar_map_fn, x_in, infer_shape=True, name='y_bar_x', parallel_iterations=4)
            print('\t y_bar_x:', self.y_bar_x.get_shape())

            ## Get latent codes for all x_i
            self.z_i_map_fn = lambda x: self.encoder.model(x, return_mode='z_i', reuse=True)
            self.z_i_x = tf.map_fn(self.z_i_map_fn, x_in, infer_shape=True, name='z_i_x', parallel_iterations=4)
            print('\t z_i_x:', self.z_i_x.get_shape())

            ## Initialize the generator
            self.x_i_hat = self.generator.model(self.z_individual)
            self.x_hat_map_fn = lambda z: self.generator.model(z, reuse=True)
            self.x_hat_logit = tf.map_fn(self.x_hat_map_fn, self.z_i_x, infer_shape=True, name='x_hat_logit', parallel_iterations=4)
            print('\t x_hat_logit:', self.x_hat_logit.get_shape())

            batch_size = tf.shape(x_in)[0]
            samples = tf.shape(x_in)[1]
            self.x_in_all = tf.reshape(x_in, [batch_size*samples]+self.x_dim)
            print('x_in_all', self.x_in_all.get_shape())
            self.x_hat_all = tf.reshape(self.x_hat_logit, [batch_size*samples]+self.x_dim)
            print('x_hat_all', self.x_in_all.get_shape())


    def _loss_op(self):
        ## Classification loss and accuracy
        # self.y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=self.y_bar_x, labels=self.y_in), name='loss')
        self.y_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y_bar_x, labels=self.y_in)
        print('y_loss', self.y_loss.get_shape())
        self.y_in_argmax = tf.argmax(self.y_in, axis=1)
        self.y_hat_argmax = tf.argmax(self.y_bar_x, axis=1)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.y_in_argmax, self.y_hat_argmax), tf.float32))

        ## Reconstruction loss and accuracy
        # self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.x_hat_logit,
        #     labels=self.x_in), axis=[2,3,4])
        # print 'recon_loss', self.recon_loss.get_shape()
        # self.recon_loss = tf.reduce_mean(self.recon_loss, axis=1)
        self.recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.x_hat_logit,
            labels=self.x_in), axis=[1,2,3,4])
        print('recon_loss', self.recon_loss.get_shape())

        self.loss = tf.reduce_mean(self.y_loss + self.recon_loss)


    def _training_ops(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.train_op = [self.optimizer.minimize(self.loss)]

        self.recon_train_op = self.optimizer.minimize(self.recon_loss)
        self.y_train_op = self.optimizer.minimize(self.y_loss)
        self.train_op = [self.recon_train_op, self.y_train_op]



    def train_step(self):
        self.global_step += 1
        batch_x, batch_y = next(self.dataset.iterator)
        feed_dict = {self.x_in: batch_x, self.y_in: batch_y}
        summary_str = self.sess.run([self.summary_op]+self.train_op,
            feed_dict=feed_dict)[-0]

        self.summary_writer.add_summary(summary_str, self.global_step)


    def test(self, test_dataset):
        batch_x, batch_y = next(test_dataset.iterator)
        feed_dict = {self.x_in: batch_x, self.y_in: batch_y}
        summary_str, accuracy = self.sess.run(
            [self.test_summary_op, self.accuracy],
            feed_dict=feed_dict)

        self.summary_writer.add_summary(summary_str, self.global_step)

        return accuracy


    def _summary_ops(self):
        with tf.variable_scope('gradients_summary'):
            if self.summarize_grads:
                self.summary_gradient_list = []
                grads = tf.gradients(self.loss, tf.trainable_variables())
                grads = list(zip(grads, tf.trainable_variables()))
                for grad, var in grads:
                    # print '{} / {}'.format(grad, var)
                    self.summary_gradient_list.append(
                        tf.summary.histogram(var.name + '/gradient', grad))

        with tf.variable_scope('variables_summary'):
            if self.summarize_vars:
                self.summary_variable_list = []
                variables = tf.trainable_variables()
                for variable in variables:
                    self.summary_variable_list.append(
                        tf.summary.histogram(variable.name + '/variable', variable))

        with tf.variable_scope('losses'):
            # self.loss_sum = tf.summary.scalar('y_loss', self.y_loss)
            # self.rec_loss_sum = tf.summary.scalar('rec_loss', self.recon_loss)
            self.loss_sum = tf.summary.scalar('y_loss', tf.reduce_mean(self.y_loss))
            self.rec_loss_sum = tf.summary.scalar('rec_loss', tf.reduce_mean(self.recon_loss))
            self.accuracy_sum = tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('endpoints'):
            self.y_hat_summary = tf.summary.histogram('y_bar_x', tf.nn.sigmoid(self.y_bar_x))
            self.y_in_summary = tf.summary.histogram('y_in', self.y_in)
            self.z_i_x_summary = tf.summary.histogram('z_i_x', self.z_i_x)

        with tf.variable_scope('images'):
            self.x_in_summary = tf.summary.image('x_in', self.x_in_all)
            self.x_hat_summary = tf.summary.image('x_hat', tf.nn.sigmoid(self.x_hat_all))

        self.summary_op = tf.summary.merge_all()

        with tf.variable_scope('test'):
            self.test_accuracy_sum = tf.summary.scalar('accuracy_test', self.accuracy)
            self.test_loss_sum = tf.summary.scalar('loss_test', tf.reduce_mean(self.y_loss))
            self.test_summary_op = tf.summary.merge([self.test_accuracy_sum,
                self.test_loss_sum])
