import tensorflow as tf
from segmentation_basemodel import SegmentationBaseModel
from ..utilities.ops import *


"""
More downsampling and deeper
"""
class ResNetBottleneck(SegmentationBaseModel):
    base_defaults={
        'conv_kernels': [64, 128, 128, 128],
        'deconv_kernels': [64, 128, 128, 128],
        'k_size': 3,
        'name': 'resnet',
    }

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        super(ResNetBottleneck, self).__init__(**self.base_defaults)

        assert self.n_classes is not None

    """
    Each resnet block applies a nonlinearity convolution to tensor_in, F(x)
    Then adds x to F(x) : F(x)+x
    and applies a nonlinearity:

    x_2 = nonlin(F(x) + x)

    Repeat for N times before returning x_n
    """
    def _encode_block(self, tensor_in, kernels, k_size, selu, name_scope):
        pass

    def _decode_block(self, tensor_in, kernels, k_size, selu, name_scope):
        pass

    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'Resnet Model'
        k_size = self.k_size
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            p0 = nonlin(conv(x_in, self.conv_kernels[0], stride=2, k_size=7, var_scope='p0', selu=1))
            p0_pool = tf.nn.max_pool(p0, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool_p0')

            r0_0_0 = nonlin(conv(p0_pool, self.conv_kernels[0], stride=1, k_size=self.k_size, var_scope='r0_0_0', selu=1))
            r0_0_1 = conv(r0_0_0, self.conv_kernels[0], stride=1, k_size=self.k_size, var_scope='r0_0_1', selu=1)
            r0_0 = nonlin(p0_pool + r0_0_1)
            r0_1_0 = nonlin(conv(r0_0, self.conv_kernels[0], stride=1, k_size=self.k_size, var_scope='r0_1_0', selu=1))
            r0_1_1 = conv(r0_1_0, self.conv_kernels[0], stride=1, k_size=self.k_size, var_scope='r0_1_1', selu=1)
            r0 = nonlin(r0_0 + r0_1_1)
            r0 = tf.contrib.nn.alpha_dropout(r0, keep_prob=keep_prob)
            r0_residual = conv(r0, self.conv_kernels[1], stride=2, k_size=1, var_scope='r0_residual')
            print '\t r0_residual', r0_residual.get_shape()

            r1_0_0 = nonlin(conv(r0, self.conv_kernels[1], stride=2, k_size=self.k_size, var_scope='r1_0_0', selu=1))
            r1_0_1 = conv(r1_0_0, self.conv_kernels[1], stride=1, k_size=self.k_size, var_scope='r1_0_1', selu=1)
            r1_0 = nonlin(r0_residual + r1_0_1)
            r1_1_0 = nonlin(conv(r1_0, self.conv_kernels[1], stride=1, k_size=self.k_size, var_scope='r1_1_0', selu=1))
            r1_1_1 = conv(r1_1_0, self.conv_kernels[1], stride=1, k_size=self.k_size, var_scope='r1_1_1', selu=1)
            r1 = nonlin(r1_0 + r1_1_1)
            r1 = tf.contrib.nn.alpha_dropout(r1, keep_prob=keep_prob)
            r1_residual = conv(r1, self.conv_kernels[2], stride=2, k_size=1, var_scope='r1_residual')
            print '\t r1_residual', r1_residual.get_shape()

            r2_0_0 = nonlin(conv(r1, self.conv_kernels[2], stride=2, k_size=self.k_size, var_scope='r2_0_0', selu=1))
            r2_0_1 = conv(r2_0_0, self.conv_kernels[2], stride=1, k_size=self.k_size, var_scope='r2_0_1', selu=1)
            r2_0 = nonlin(r1_residual + r2_0_1)
            r2_1_0 = nonlin(conv(r2_0, self.conv_kernels[2], stride=1, k_size=self.k_size, var_scope='r2_1_0', selu=1))
            r2_1_1 = conv(r2_1_0, self.conv_kernels[2], stride=1, k_size=self.k_size, var_scope='r2_1_1', selu=1)
            r2 = nonlin(r2_0 + r2_1_1)
            r2 = tf.contrib.nn.alpha_dropout(r2, keep_prob=keep_prob)
            r2_residual = conv(r2, self.conv_kernels[3], stride=2, k_size=1, var_scope='r2_residual')
            print '\t r2_residual', r2_residual.get_shape()

            r3_0_0 = nonlin(conv(r2, self.conv_kernels[3], stride=2, k_size=self.k_size, var_scope='r3_0_0', selu=1))
            r3_0_1 = conv(r3_0_0, self.conv_kernels[3], stride=1, k_size=self.k_size, var_scope='r3_0_1', selu=1)
            r3_0 = nonlin(r2_residual + r3_0_1)
            r3_1_0 = nonlin(conv(r3_0, self.conv_kernels[3], stride=1, k_size=self.k_size, var_scope='r3_1_0', selu=1))
            r3_1_1 = conv(r3_1_0, self.conv_kernels[3], stride=1, k_size=self.k_size, var_scope='r3_1_1', selu=1)
            r3 = nonlin(r3_0 + r3_1_1)
            r3 = tf.contrib.nn.alpha_dropout(r3, keep_prob=keep_prob)
            r3_residual = deconv(r3, self.deconv_kernels[3], upsample_rate=2, k_size=1, var_scope='r3_residual')
            print '\t r3_residual', r3_residual.get_shape()

            d3_0_0 = nonlin(deconv(r3, self.deconv_kernels[3], upsample_rate=2, k_size=self.k_size, var_scope='d3_0_0', selu=1))
            d3_0_1 = conv(d3_0_0, self.deconv_kernels[3], stride=1, k_size=self.k_size, var_scope='d3_0_1', selu=1)
            d3_0 = nonlin(r3_residual + d3_0_1)
            d3_1_0 = nonlin(conv(d3_0, self.deconv_kernels[3], stride=1, k_size=self.k_size, var_scope='d3_1_0', selu=1))
            d3_1_1 = conv(d3_1_0, self.deconv_kernels[3], stride=1, k_size=self.k_size, var_scope='d3_1_1', selu=1)
            d3 = nonlin(d3_0 + d3_1_1)
            d3 = tf.contrib.nn.alpha_dropout(d3, keep_prob=keep_prob)
            d3_residual = deconv(d3, self.deconv_kernels[2], upsample_rate=2, k_size=1, var_scope='d2_residual')
            print '\t d3_residual', d3_residual.get_shape()

            d2_0_0 = nonlin(deconv(d3, self.deconv_kernels[2], upsample_rate=2, k_size=self.k_size, var_scope='d2_0_0', selu=1))
            d2_0_1 = conv(d2_0_0, self.deconv_kernels[2], stride=1, k_size=self.k_size, var_scope='d2_0_1', selu=1)
            # d2_0 = nonlin(r2_residual + d2_0_1) ## skip connection
            d2_0 = nonlin(d3_residual + d2_0_1)
            d2_1_0 = nonlin(conv(d2_0, self.deconv_kernels[2], stride=1, k_size=self.k_size, var_scope='d2_1_0', selu=1))
            d2_1_1 = conv(d2_1_0, self.deconv_kernels[2], stride=1, k_size=self.k_size, var_scope='d2_1_1', selu=1)
            d2 = nonlin(d2_0 + d2_1_1)
            d2 = tf.contrib.nn.alpha_dropout(d2, keep_prob=keep_prob)
            d2_residual = deconv(d2, self.deconv_kernels[1], upsample_rate=2, k_size=1, var_scope='d2_residual')
            print '\t d2_residual', d2_residual.get_shape()

            d1_0_0 = nonlin(deconv(d2, self.deconv_kernels[1], upsample_rate=2, k_size=self.k_size, var_scope='d1_0_0', selu=1))
            d1_0_1 = conv(d1_0_0, self.deconv_kernels[1], stride=1, k_size=self.k_size, var_scope='d1_0_1', selu=1)
            d1_0 = nonlin(d2_residual + d1_0_1)
            d1_1_0 = nonlin(conv(d1_0, self.deconv_kernels[1], stride=1, k_size=self.k_size, var_scope='d1_1_0', selu=1))
            d1_1_1 = conv(d1_1_0, self.deconv_kernels[1], stride=1, k_size=self.k_size, var_scope='d1_1_1', selu=1)
            d1 = nonlin(d1_0 + d1_1_1)
            d1 = tf.contrib.nn.alpha_dropout(d1, keep_prob=keep_prob)
            d1_residual = deconv(d1, self.deconv_kernels[0], upsample_rate=2, k_size=1, var_scope='d1_residual')
            print '\t d1_residual', d1_residual.get_shape()

            d0_0_0 = nonlin(deconv(d1, self.deconv_kernels[0], upsample_rate=2, k_size=self.k_size, var_scope='d0_0_0', selu=1))
            d0_0_1 = conv(d0_0_0, self.deconv_kernels[0], stride=1, k_size=self.k_size, var_scope='d0_0_1', selu=1)
            d0_0 = nonlin(d1_residual + d0_0_1)
            d0_1_0 = nonlin(conv(d0_0, self.deconv_kernels[0], stride=1, k_size=self.k_size, var_scope='d0_1_0', selu=1))
            d0_1_1 = conv(d0_1_0, self.deconv_kernels[0], stride=1, k_size=self.k_size, var_scope='d0_1_1', selu=1)
            d0 = nonlin(d0_0 + d0_1_1)
            print '\t d0', d0.get_shape()

            y_hat = deconv(d0, self.n_classes, upsample_rate=2, k_size=5, var_scope='y_hat')

            return y_hat



class ResNetBottleneckTraining(ResNetBottleneck):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(ResNetBottleneckTraining, self).__init__(**self.train_defaults)


class ResNetBottleneckInference(ResNetBottleneck):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(ResNetBottleneckInference, self).__init__(**self.inference_defaults)
