from __future__ import print_function
import tensorflow as tf
from .segmentation_basemodel import Segmentation
from ..utilities.ops import *

class FCN(Segmentation):
    def __init__(self, **kwargs):
        fcn_defaults={
            'k_size': [7,5,3],
            'conv_kernels': None,
            'name': 'fcn',
            'snapshot_name': 'fcn'}

        fcn_defaults.update(**kwargs)
        super(FCN, self).__init__(**fcn_defaults)

        assert self.n_classes is not None
        assert self.conv_kernels is not None

    ## Layer flow copied from:
    ## https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py
    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print('FCN Model')
        k_size = self.k_size
        nonlin = self.nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print('\t x_in', x_in.get_shape())

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=k_size[0], stride=1, var_scope='c0_0'))
            c0_1 = nonlin(conv(c0_0, self.conv_kernels[0], k_size=k_size[0], stride=1, var_scope='c0_1'))
            c0_pool = tf.nn.max_pool(c0_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c0_pool')

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=k_size[1], stride=1, var_scope='c1_0'))
            c1_1 = nonlin(conv(c1_0, self.conv_kernels[1], k_size=k_size[1], stride=1, var_scope='c1_1'))
            c1_pool = tf.nn.max_pool(c1_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c1_pool')

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=k_size[2], stride=1, var_scope='c2_0'))
            c2_1 = nonlin(conv(c2_0, self.conv_kernels[2], k_size=k_size[2], stride=1, var_scope='c2_1'))
            c2_1 = tf.contrib.nn.alpha_dropout(c2_1, keep_prob=keep_prob)
            c2_pool = tf.nn.max_pool(c2_1, [1,4,4,1], [1,4,4,1], padding='VALID',
                name='c2_pool')

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=k_size[3], stride=1, var_scope='c3_0'))
            c3_1 = nonlin(conv(c3_0, self.conv_kernels[3], k_size=k_size[3], stride=1, var_scope='c3_1'))
            c3_1 = tf.contrib.nn.alpha_dropout(c3_1, keep_prob=keep_prob)
            c3_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c3_pool')

            ## Alternate layer connectivity
            prediction_3 = nonlin(conv(c3_pool, self.n_classes, stride=1, var_scope='pred3'))
            prediction_2 = nonlin(conv(c2_pool, self.n_classes, stride=1, var_scope='pred2'))
            prediction_1 = nonlin(conv(c1_pool, self.n_classes, stride=1, var_scope='pred1'))

            upscore3 = nonlin(deconv(prediction_3, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups3'))
            upscore3 = upscore3 + prediction_2
            upscore3_ups = nonlin(deconv(upscore3, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups3_ups'))

            upscore2 = nonlin(deconv(prediction_2, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups2'))
            upscore2 = upscore2 + upscore3_ups
            upscore2_ups = nonlin(deconv(upscore2, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups2_ups'))
            upscore2 = prediction_1 + upscore2_ups

            preout = nonlin(deconv(upscore2, self.n_classes, k_size=4, upsample_rate=4, var_scope='preout'))

            y_hat = conv(preout, self.n_classes, k_size=3, stride=1, var_scope='y_hat')

            return y_hat


class FCNTraining(FCN):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(FCNTraining, self).__init__(**self.train_defaults)


class FCNInference(FCN):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(FCNInference, self).__init__(**self.inference_defaults)
