import tensorflow as tf
from segmentation_basemodel import SegmentationBaseModel
from ..utilities.ops import *

class VGG(SegmentationBaseModel):
    vgg_defaults={
        'name': 'vgg',
        'snapshot_name': 'vgg'}

    def __init__(self, **kwargs):
        self.vgg_defaults.update(**kwargs)
        super(VGG, self).__init__(**self.vgg_defaults)

        assert self.n_classes is not None
        if self.mode=='TRAIN': assert self.dataset.dstype=='ImageMask'


    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'VGG-FCN Model'
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=3, stride=1, var_scope='c0_0'))
            c0_1 = nonlin(conv(c0_0, self.conv_kernels[0], k_size=3, stride=1, var_scope='c0_1'))
            # c0_1 = batch_norm(c0_1, reuse=reuse, training=training, var_scope='c0_1_bn')
            c0_pool = tf.nn.max_pool(c0_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c0_pool')
            print '\t c0_pool', c0_pool.get_shape() ## 128

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=3, stride=1, var_scope='c1_0'))
            c1_1 = nonlin(conv(c1_0, self.conv_kernels[1], k_size=3, stride=1, var_scope='c1_1'))
            # c1_1 = batch_norm(c1_1, training=training, var_scope='c1_1_bn')
            c1_pool = tf.nn.max_pool(c1_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c1_pool')
            print '\t c1_pool', c1_pool.get_shape() ## 64

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=3, stride=1, var_scope='c2_0'))
            c2_1 = nonlin(conv(c2_0, self.conv_kernels[2], k_size=3, stride=1, var_scope='c2_1'))
            # c2_1 = batch_norm(c2_1, training=training, var_scope='c2_1_bn')
            c2_pool = tf.nn.max_pool(c2_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c2_pool')
            print '\t c2_pool', c2_pool.get_shape() ## 32

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=3, stride=1, var_scope='c3_0'))
            # c3_0 = tf.nn.dropout(c3_0, keep_prob=keep_prob)
            c3_0 = tf.contrib.nn.alpha_dropout(c3_0, keep_prob=keep_prob)
            c3_1 = nonlin(conv(c3_0, self.conv_kernels[3], k_size=3, stride=1, var_scope='c3_1'))
            # c3_1 = batch_norm(c3_1, training=training, var_scope='c3_1_bn')
            c3_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c3_pool')
            print '\t c3_pool', c3_pool.get_shape()  ## inputs / 16 = 16

            d1 = nonlin(deconv(c3_pool, self.deconv_kernels[1], upsample_rate=4, var_scope='d1'))
            d1 = nonlin(conv(d1, self.deconv_kernels[1], stride=1, var_scope='dc1'))
            # d1 = batch_norm(d1, reuse=reuse, training=training, var_scope='d1_bn')
            print '\t d1', d1.get_shape() ## 16*4 = 64

            d0 = nonlin(deconv(d1, self.deconv_kernels[0], var_scope='d0'))
            d0 = nonlin(conv(d0, self.deconv_kernels[0], stride=1, var_scope='dc0'))
            # d0 = batch_norm(d0, training=training, var_scope='d0_bn')
            print '\t d0', d0.get_shape() ## 64*2 = 128

            y_hat = deconv(d0, self.n_classes, var_scope='y_hat')
            print '\t y_hat', y_hat.get_shape() ## 128*2 = 256

            return y_hat



class VGGTraining(VGG):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(VGGTraining, self).__init__(**self.train_defaults)


class VGGInference(VGG):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(VGGInference, self).__init__(**self.inference_defaults)
