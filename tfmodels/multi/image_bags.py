import tensorflow as tf
import sys

from ..utilities.ops import *
from ..generative.encoder_basemodel import BaseEncoder
from ..utilities.basemodel import BaseModel

"""
DeepSets / Multi-instance / repeated measures

Multi instance models are like this:
we have sets, X ~ [{x1, x2, x3, ..., xn}_1,
                  [{x1, x2, x3, ..., xn}_2, ...]

we have 1 label, Y, for each set

the goal is to predict Y from the elements of X

for X in minibatch:
    sample an X
    for x in X:
        encode x --> z
    apply permutation invariant combination:
    \hat{z} --> mean({z})

fit a classifier: p(Y|\hat{z})

it should accept a dataset that produces bagged observations
these datasets have an op, dataset.bag_op that returns an (n+1)D minibatch:

if 1D vectors (i.e. MNIST, CSV): [batch_size, observations, dimension]
if 3D images: [batch_size, observations, h, w, c]

TODO: sparse tensors along DIM 1 to support variable number of observations


For BaggedMNIST:

in BaggedMNIST we have the mnist digits, and a positive class

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
class FCEncoder(BaseEncoder):
    fc_encoder_defaults = {
        'hidden_dim': [512, 512],
        'name': 'FCEncoder'
        }

    def __init__(self, name='linear_encoder', **kwargs):
        self.fc_encoder_defaults.update(**kwargs)
        super(FCEncoder, self).__init__(**self.fc_encoder_defaults)

        self.name = name

    def model(self, x_in, keep_prob=0.5, reuse=False, return_z=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up FCEncoder model'
            print '\t x_in: ', x_in.get_shape()
            nonlin = self.nonlin
            h0 = nonlin(linear(x_in, self.hidden_dim[0], var_scope='e_h0'))
            h1 = nonlin(linear(h0, self.hidden_dim[1], var_scope='e_h1'))
            self.zed_x = linear(h1, self.z_dim, var_scope='zed')

            ## Reduce mean
            self.zed_x_bar = tf.reduce_mean(self.zed_x, axis=0)

            if return_z:
                return self.zed_x_bar, self.zed_x
            else:
                return self.zed_x_bar


"""
Take normal [batch_size, --dimensions--]
encode them, then squash the batch by taking a mean
"""
class ConvEncoder(BaseEncoder):
    conv_encoder_defaults = {
        'kernels': [64, 128],
        'hidden_dim': [512],
        'name': 'ConvEncoder',
        }

    def __init__(self, name='convolutional_encoder', **kwargs):
        self.conv_encoder_defaults.update(**kwargs)
        super(ConvEncoder, self).__init__(**self.conv_encoder_defaults)

        self.name = name

    def model(self, x_in, keep_prob=0.5, reuse=False, return_z=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            print 'Setting up ConvEncoder model'
            print '\t x_in: ', x_in.get_shape()
            nonlin = tf.nn.selu

            c0 = nonlin(conv(x_in, self.kernels[0], k_size=5, stride=2, var_scope='c0'))
            c0_p = tf.nn.max_pool(c0, [1,2,2,1], [1,2,2,1], padding='VALID')
            c1 = nonlin(conv(c0_p, self.kernels[1], k_size=3, stride=2, var_scope='c1'))
            c1_flat = tf.contrib.layers.flatten(c1)

            h0 = nonlin(linear(c1_flat, self.hidden_dim[0], var_scope='h0'))

            self.zed_x = linear(h0, self.z_dim, var_scope='zed')

            ## Reduce mean
            self.zed_x_bar = tf.reduce_mean(self.zed_x, axis=0)

            if return_z:
                return self.zed_x_bar, self.zed_x
            else:
                return self.zed_x_bar


class ImageBagModel(BaseModel):
    image_bag_default = {
        'encoder_type': 'CONV', ## CONV or DENSE
        'learning_rate': 1e-4,
        'n_classes': 2,
        'name': 'deepsets',
        'no_classifier': True,
        'sess': None,
        'summarize_grads': False,
        'summarize_vars': False,
        'x_dim': [28*28],
        'z_dim': 32
        }

    def __init__(self, name='deepsets', **kwargs):
        self.image_bag_default.update(**kwargs)
        super(ImageBagModel, self).__init__(**self.image_bag_default)

        self.name = name
        assert self.sess is not None

        ## first two dimensions of x_in should be None
        if self.encoder_type == 'CONV':
            assert len(self.x_dim) == 3
            self.x_individual = tf.placeholder('float', shape=[None]+self.x_dim, name='x_individual')
            self.x_in = tf.placeholder('float', shape=[None, None]+self.x_dim)
            encoder_class = ConvEncoder
        elif self.encoder_type == 'DENSE':
            self.x_individual = tf.placeholder('float', shape=[None, self.x_dim[0]], name='x_individual')
            self.x_in = tf.placeholder('float', shape=[None, None, self.x_dim[0]])
            encoder_class = FCEncoder

        self.y_in = tf.placeholder('float', shape=[None, self.n_classes], name='y_in')

        ## use n_classes to get a discriminiator trained to detect positive x_i

        if self.no_classifier:
            self.encoder = encoder_class(z_dim=self.n_classes)
        else:
            self.encoder = encoder_class(z_dim=self.z_dim)

        self.encoder.print_info()

        self.y_hat = self.model(self.x_in)

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
        print '\t Setting up Classification model'
        print '\t Using name scope: {}'.format(self.name)
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            nonlin = self.nonlin

            ## Initialize the encoder
            dummy_z, self.z_individual = self.encoder.model(self.x_individual, return_z=True)

            print '\t x_in:', x_in.get_shape()
            self.encode_map_fn = lambda x: self.encoder.model(x, reuse=True)
            self.z_hat = tf.map_fn(self.encode_map_fn, x_in, infer_shape=False, name='z_hat')
            print '\t z_hat:', self.z_hat.get_shape() ## batch_size, dimensions

            ## Use the output of z_hat directly; requires z_dim=self.n_classes
            if self.no_classifier:
                y_hat = tf.identity(self.z_hat, name='y_hat')
            ## Classifier
            else:
                h0 = nonlin(linear(z_hat, 2, selu=1, var_scope='h0'))
                h1 = nonlin(linear(h0, 256, selu=1, var_scope='h1'))
                y_hat = linear(self.z_hat, self.n_classes, selu=1, var_scope='y_hat')

        return y_hat


    def _loss_op(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.y_hat, labels=self.y_in), name='loss')
        self.y_in_argmax = tf.argmax(self.y_in, axis=1)
        self.y_hat_argmax = tf.argmax(self.y_hat, axis=1)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            self.y_in_argmax, self.y_hat_argmax), tf.float32))


    def _training_ops(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)


    def train_step(self):
        self.global_step += 1
        batch_x, batch_y = next(self.dataset.iterator)
        feed_dict = {self.x_in: batch_x, self.y_in: batch_y}
        summary_str = self.sess.run([self.train_op, self.summary_op], feed_dict=feed_dict)[-1]

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
        with tf.variable_scope('summary'):
            if self.summarize_grads:
                self.summary_gradient_list = []
                grads = tf.gradients(self.loss, tf.trainable_variables())
                grads = list(zip(grads, tf.trainable_variables()))
                for grad, var in grads:
                    # print '{} / {}'.format(grad, var)
                    self.summary_gradient_list.append(
                        tf.summary.histogram(var.name + '/gradient', grad))

            if self.summarize_vars:
                self.summary_variable_list = []
                variables = tf.trainable_variables()
                for variable in variables:
                    self.summary_variable_list.append(
                        tf.summary.histogram(var.name + '/variable', variable))

            self.loss_sum = tf.summary.scalar('loss', self.loss)
            self.accuracy_sum = tf.summary.scalar('accuracy', self.accuracy)

            self.y_hat_summary = tf.summary.histogram('y_hat', tf.nn.sigmoid(self.y_hat))
            self.y_in_summary = tf.summary.histogram('y_in', self.y_in)
            self.z_hat_summary = tf.summary.histogram('z_hat', self.z_hat)

            self.summary_op = tf.summary.merge_all()

            self.test_accuracy_sum = tf.summary.scalar('accuracy_test', self.accuracy)
            self.test_loss_sum = tf.summary.scalar('loss_test', self.loss)
            self.test_summary_op = tf.summary.merge([self.test_accuracy_sum,
                self.test_loss_sum])
