import tensorflow as tf
from segmentation_basemodel import SegmentationBaseModel
from ..utilities.ops import *

class SegNet(SegmentationBaseModel):
    base_defaults={
        'name': 'segnet',
        'k_size': 5,
        'snapshot_name': 'segnet'
    }

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        super(SegNet, self).__init__(**self.base_defaults)

        assert self.n_classes is not None
        if self.mode=='TRAIN': assert self.dataset.dstype=='ImageMask'

    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'SegNet Model'
        k_size = self.k_size
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=k_size, stride=1, var_scope='c0_0'))
            c0_1 = nonlin(conv(c0_0, self.conv_kernels[0], k_size=k_size, stride=1, var_scope='c0_1'))
            c0_pool, c0_max = tf.nn.max_pool_with_argmax(c0_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c0_pool')
            print '\t c0_pool', c0_pool.get_shape() ## 128

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=k_size, stride=1, var_scope='c1_0'))
            c1_1 = nonlin(conv(c1_0, self.conv_kernels[1], k_size=k_size, stride=1, var_scope='c1_1'))
            c1_1 = tf.contrib.nn.alpha_dropout(c1_1, keep_prob=keep_prob, name='c1_1_do')
            c1_pool, c1_max = tf.nn.max_pool_with_argmax(c1_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c1_pool')
            print '\t c1_pool', c1_pool.get_shape() ## 64

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=k_size, stride=1, var_scope='c2_0'))
            c2_1 = nonlin(conv(c2_0, self.conv_kernels[2], k_size=k_size, stride=1, var_scope='c2_1'))
            c2_1 = tf.contrib.nn.alpha_dropout(c2_1, keep_prob=keep_prob, name='c2_1_do')
            c2_pool, c2_max = tf.nn.max_pool_with_argmax(c2_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c2_pool')
            print '\t c2_pool', c2_pool.get_shape() ## 32

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=k_size, stride=1, var_scope='c3_0'))
            c3_1 = nonlin(conv(c3_0, self.conv_kernels[3], k_size=k_size, stride=1, var_scope='c3_1'))
            c3_1 = tf.contrib.nn.alpha_dropout(c3_1, keep_prob=keep_prob)
            c3_pool, c3_max = tf.nn.max_pool_with_argmax(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID', name='c3_pool')
            print '\t c3_pool', c3_pool.get_shape()  ## inputs / 16 = 16

            ## Unpool instead of deconvolution
            d2 = unpool(c3_pool, c3_max, k_size=[1,2,2,1], var_scope='unpool3')
            d2 = nonlin(conv(d2, self.deconv_kernels[2], stride=1, var_scope='dc2'))
            d2 = tf.contrib.nn.alpha_dropout(d2, keep_prob=keep_prob)
            print '\t d2', d2.get_shape() ## 16*2 = 32

            d1 = unpool(d2, c2_max, k_size=[1,2,2,1], var_scope='unpool2')
            d1 = nonlin(conv(d1, self.deconv_kernels[1], stride=1, var_scope='dc1'))
            d1 = tf.contrib.nn.alpha_dropout(d1, keep_prob=keep_prob)
            print '\t d1', d1.get_shape() ## 32*2 = 64

            d0 = unpool(d1, c1_max, k_size=[1,2,2,1], var_scope='unpool1')
            d0 = nonlin(conv(d0, self.deconv_kernels[0], stride=1, var_scope='dc0'))
            print '\t d0', d0.get_shape() ## 64*2 = 128

            # y_hat = unpool(d0, c0_max, k_size=[1,2,2,1], var_scope='unpool0')
            # y_hat = conv(y_hat, self.n_classes, stride=1, pad='SAME', var_scope='y_hat')
            y_hat = nonlin(deconv(d0, self.n_classes, var_scope='y_hat_0'))
            y_hat = conv(y_hat, self.n_classes, stride=1, pad='SAME', var_scope='y_hat')
            print '\t y_hat', y_hat.get_shape() ## 128*2 = 256

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
