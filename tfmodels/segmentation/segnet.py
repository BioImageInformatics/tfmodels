from __future__ import print_function
import tensorflow as tf
from .segmentation_basemodel import Segmentation
from ..utilities.ops import *

class SegNet(Segmentation):

    def __init__(self, **kwargs):
        segnet_defaults={
            'conv_kernels': None,
            'deconv_kernels': None,
            'k_size': 3,
            'name': 'segnet',
        }
        segnet_defaults.update(**kwargs)

        assert segnet_defaults['n_classes'] is not None
        assert segnet_defaults['conv_kernels'] is not None
        assert segnet_defaults['deconv_kernels'] is not None

        super(SegNet, self).__init__(**segnet_defaults)


    ## TODO rewrite programatically for variable length conv/deconv
    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print('SegNet Model')
        k_size = self.k_size
        nonlin = self.nonlin
        print('Non-linearity:', nonlin)

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print('\t x_in', x_in.get_shape())
            conv_args = {'k_size': k_size, 'stride': 1, 'selu': True}

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=k_size, stride=1, var_scope='c0_0', selu=1))
            c0_1 = nonlin(conv(c0_0, self.conv_kernels[0], k_size=k_size, stride=1, var_scope='c0_1', selu=1))
            c0_pool, c0_max = tf.nn.max_pool_with_argmax(c0_1, [1,2,2,1], [1,2,2,1], padding='VALID', name='c0_pool')
            ## 64

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=k_size, stride=1, var_scope='c1_0', selu=1))
            c1_1 = nonlin(conv(c1_0, self.conv_kernels[1], k_size=k_size, stride=1, var_scope='c1_1', selu=1))
            # c1_1 = tf.contrib.nn.alpha_dropout(c1_1, keep_prob=keep_prob, name='c1_1_do')
            c1_pool, c1_max = tf.nn.max_pool_with_argmax(c1_1, [1,2,2,1], [1,2,2,1], padding='VALID', name='c1_pool')
            ## 32

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=k_size, stride=1, var_scope='c2_0', selu=1))
            c2_1 = nonlin(conv(c2_0, self.conv_kernels[2], k_size=k_size, stride=1, var_scope='c2_1', selu=1))
            c2_2 = nonlin(conv(c2_1, self.conv_kernels[2], k_size=k_size, stride=1, var_scope='c2_2', selu=1))
            # c2_2 = tf.contrib.nn.alpha_dropout(c2_2, keep_prob=keep_prob, name='c2_1_do')
            c2_pool, c2_max = tf.nn.max_pool_with_argmax(c2_2, [1,2,2,1], [1,2,2,1], padding='VALID', name='c2_pool')
            ## 16

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=k_size, stride=1, var_scope='c3_0', selu=1))
            c3_1 = nonlin(conv(c3_0, self.conv_kernels[3], k_size=k_size, stride=1, var_scope='c3_1', selu=1))
            c3_2 = nonlin(conv(c3_1, self.conv_kernels[3], k_size=k_size, stride=1, var_scope='c3_2', selu=1))
            # c3_2 = tf.contrib.nn.alpha_dropout(c3_2, keep_prob=keep_prob)
            c3_pool, c3_max = tf.nn.max_pool_with_argmax(c3_2, [1,2,2,1], [1,2,2,1], padding='VALID', name='c3_pool')
            ## 8

            c4_0 = nonlin(conv(c3_pool, self.conv_kernels[4], k_size=k_size, stride=1, var_scope='c4_0', selu=1))
            c4_1 = nonlin(conv(c4_0, self.conv_kernels[4], k_size=k_size, stride=1, var_scope='c4_1', selu=1))
            c4_2 = nonlin(conv(c4_1, self.conv_kernels[4], k_size=k_size, stride=1, var_scope='c4_2', selu=1))
            c4_2 = tf.contrib.nn.alpha_dropout(c4_2, keep_prob=keep_prob)
            c4_pool, c4_max = tf.nn.max_pool_with_argmax(c4_2, [1,2,2,1], [1,2,2,1], padding='VALID', name='c4_pool')
            ## 4

            ## Unpool instead of deconvolution
            unpool4 = unpool(c4_pool, c4_max, k_size=[1,2,2,1], var_scope='unpool4')
            d4_0 = nonlin(conv(unpool4, self.deconv_kernels[4], k_size=k_size, stride=1, var_scope='d4_0', selu=1))
            d4_1 = nonlin(conv(d4_0, self.deconv_kernels[4], k_size=k_size, stride=1, var_scope='d4_1', selu=1))
            d4 = nonlin(conv(d4_1, self.deconv_kernels[3], k_size=k_size, stride=1, var_scope='d4', selu=1))
            d4 = tf.contrib.nn.alpha_dropout(d4, keep_prob=keep_prob)

            unpool3 = unpool(d4, c3_max, k_size=[1,2,2,1], var_scope='unpool3')
            d3_0 = nonlin(conv(unpool3, self.deconv_kernels[3], k_size=k_size, stride=1, var_scope='d3_0', selu=1))
            d3_1 = nonlin(conv(d3_0, self.deconv_kernels[3], k_size=k_size, stride=1, var_scope='d3_1', selu=1))
            d3 = nonlin(conv(d3_1, self.deconv_kernels[2], k_size=k_size, stride=1, var_scope='d3', selu=1))

            unpool2 = unpool(d3, c2_max, k_size=[1,2,2,1], var_scope='unpool2')
            d2_0 = nonlin(conv(unpool2, self.deconv_kernels[2], k_size=k_size, stride=1, var_scope='d2_0', selu=1))
            d2_1 = nonlin(conv(d2_0, self.deconv_kernels[2], stride=1, k_size=k_size, var_scope='d2_1', selu=1))
            d2 = nonlin(conv(d2_1, self.deconv_kernels[1], stride=1, k_size=k_size, var_scope='d2', selu=1))

            unpool1 = unpool(d2, c1_max, k_size=[1,2,2,1], var_scope='unpool1')
            d1_0 = nonlin(conv(unpool1, self.deconv_kernels[1], k_size=k_size, stride=1, var_scope='d1_0', selu=1))
            d1 = nonlin(conv(d1_0, self.deconv_kernels[0], k_size=k_size, stride=1, var_scope='d1', selu=1))

            unpool0 = unpool(d1, c0_max, k_size=[1,2,2,1], var_scope='unpool0')
            d0_0 = nonlin(conv(unpool0, self.deconv_kernels[0], k_size=k_size, stride=1, var_scope='d0_0', selu=1))
            d0 = nonlin(conv(d0_0, self.deconv_kernels[0], k_size=k_size, stride=1, var_scope='d0', selu=1))

            y_hat = nonlin(conv(d0, self.n_classes, k_size=k_size, stride=1, pad='SAME', var_scope='y_hat_0', selu=1))
            y_hat = conv(y_hat, self.n_classes, k_size=k_size, stride=1, pad='SAME', var_scope='y_hat')
            return y_hat



class SegNetTraining(SegNet):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(SegNetTraining, self).__init__(**self.train_defaults)


class SegNetInference(SegNet):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(SegNetInference, self).__init__(**self.inference_defaults)
