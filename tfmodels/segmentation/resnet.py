import tensorflow as tf
from segmentation_basemodel import SegmentationBaseModel
from ..utilities.ops import *


"""
More downsampling and deeper
"""
class ResNet(SegmentationBaseModel):
    base_defaults={
        'kernels': [64, 64, 64, 128],
        'k_size': 3,
        'modules': None,
        'name': 'resnet_module',
        'stacks': 5,
    }

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        super(ResNet, self).__init__(**self.base_defaults)

        assert self.n_classes is not None
        ## Check input shape is compatible with the number of downsampling modules
        self.modules = len(self.kernels)
        start_size = self.x_dims[0]/4 ## start with 2 stride conv and pool
        min_dimension = start_size / np.power(2,self.modules)
        print 'MINIMIUM DIMENSION: ', min_dimension
        assert min_dimension >= 1

    """
    Accept x_1

    Each resnet block applies a nonlinearity convolution to tensor_in, F(x)
    Then adds x to F(x) : F(x)+x
    and applies a nonlinearity:

    x_2 = nonlin(F(x) + x)

    Repeat for N times before returning x_n
    """
    def _residual_block(self, x_1, kernels, k_size=3, block=0, selu=1, stacks=3, name_scope='e'):
        nonlin = self.nonlin
        conv_settings = {'n_kernel': kernels, 'stride': 1, 'k_size': self.k_size, 'selu': 1}

        with tf.variable_scope('{}_{}'.format(name_scope, block)):
            for stack in xrange(stacks):
                stack_name='{}_{}_{}_'.format(name_scope, block, stack)
                with tf.variable_scope(stack_name):
                    x_s_1 = nonlin(conv(x_1, var_scope=stack_name+'0', **conv_settings))
                    x_s_2 = conv(x_s_1, var_scope=stack_name+'1', **conv_settings)
                    x_1 = nonlin(x_1 + x_s_2)

        return x_1


    # def _decode_block(self, x_1, kernels, k_size, block=0, selu=1, stacks=3, name_scope='d'):
    #     nonlin = self.nonlin
    #     deconv_settings = {'n_kernel': kernels, 'upsample_rate': 2, 'k_size': self.k_size, 'selu': 1}
    #
    #     for stack in xrange(stacks):
    #         stack_name='{}_{}_{}_'.format(name_scope, block, stack)
    #         x_s_1 = nonlin()


    """
    the _residual_block() method takes the input and builds the normal residual chain
    the output shape matches the input shape

    do the upsampling/downsampling outside of _residual_block()
    do dropout after the residual block, before the down/up sampling
    """
    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'Resnet Model'
        k_size = self.k_size
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            p0 = nonlin(conv(x_in, self.kernels[0], stride=2, k_size=7, var_scope='p0', selu=1))
            signal = tf.nn.max_pool(p0, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool_p0')
            print '\t signal', signal.get_shape()

            for block in xrange(self.modules-1):
                block_name = 'r{}_residual'.format(block)
                signal = self._residual_block(signal, self.kernels[block],
                    block=block, stacks=self.stacks, name_scope='r')
                signal = conv(signal, self.kernels[block+1], stride=2, k_size=1,
                    var_scope=block_name)
                print '\t {}'.format(block_name), signal.get_shape()

            signal = tf.contrib.nn.alpha_dropout(signal, keep_prob=keep_prob)

            for block in xrange(self.modules, 0, -1):
                block_name = 'd{}_residual'.format(block)
                signal = self._residual_block(signal, self.kernels[block],
                    block=block, stacks=self.stacks, name_scope='d')
                signal = deconv(signal, self.kernels[block-1], upsample_rate=2, k_size=1,
                    var_scope=block_name)
                print '\t {}'.format(block_name), signal.get_shape()

            # r0 = self._residual_block(p0_pool, self.kernels[0], block=0, stacks=self.stacks, name_scope='r')
            # # r0 = tf.contrib.nn.alpha_dropout(r0, keep_prob=keep_prob)
            # r0_residual = conv(r0, self.kernels[1], stride=2, k_size=1, var_scope='r0_residual')
            # print '\t r0_residual', r0_residual.get_shape()
            #
            # r1 = self._residual_block(r0_residual, self.kernels[1], block=1, stacks=self.stacks, name_scope='r')
            # # r1 = tf.contrib.nn.alpha_dropout(r1, keep_prob=keep_prob)
            # r1_residual = conv(r1, self.kernels[2], stride=2, k_size=1, var_scope='r1_residual')
            # print '\t r1_residual', r1_residual.get_shape()
            #
            # r2 = self._residual_block(r1_residual, self.kernels[2], block=2, stacks=self.stacks, name_scope='r')
            # # r2 = tf.contrib.nn.alpha_dropout(r2, keep_prob=keep_prob)
            # r2_residual = conv(r2, self.kernels[3], stride=2, k_size=1, var_scope='r2_residual')
            # print '\t r2_residual', r2_residual.get_shape()
            #
            # r3 = self._residual_block(r2_residual, self.kernels[3], block=3, stacks=self.stacks, name_scope='r')
            # r3 = tf.contrib.nn.alpha_dropout(r3, keep_prob=keep_prob)
            # r3_residual = deconv(r3, self.kernels[3], upsample_rate=2, k_size=1, var_scope='r3_residual')
            # print '\t r3_residual', r3_residual.get_shape()
            #
            # d3 = self._residual_block(r3_residual, self.kernels[3], block=3, stacks=self.stacks, name_scope='d')
            # d3 = tf.contrib.nn.alpha_dropout(d3, keep_prob=keep_prob)
            # d3_residual = deconv(d3, self.kernels[2], upsample_rate=2, k_size=1, var_scope='d3_residual')
            # print '\t d3_residual', d3_residual.get_shape()
            #
            # d2 = self._residual_block(d3_residual, self.kernels[2], block=2, stacks=self.stacks, name_scope='d')
            # # d2 = tf.contrib.nn.alpha_dropout(d2, keep_prob=keep_prob)
            # d2_residual = deconv(d2, self.kernels[1], upsample_rate=2, k_size=1, var_scope='d2_residual')
            # print '\t d2_residual', d2_residual.get_shape()
            #
            # d1 = self._residual_block(d2_residual, self.kernels[1], block=1, stacks=self.stacks, name_scope='d')
            # # d1 = tf.contrib.nn.alpha_dropout(d1, keep_prob=keep_prob)
            # d1_residual = deconv(d1, self.n_classes, upsample_rate=2, k_size=7, var_scope='d1_residual')
            # print '\t d1_residual', d1_residual.get_shape()

            y_hat = deconv(signal, self.n_classes, upsample_rate=2, k_size=3, var_scope='y_hat')
            # y_hat = conv(d1_residual, self.n_classes, stride=1, k_size=5, var_scope='y_hat')
            print '\t y_hat', y_hat.get_shape()

            return y_hat



class ResNetTraining(ResNet):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(ResNetTraining, self).__init__(**self.train_defaults)


class ResNetInference(ResNet):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(ResNetInference, self).__init__(**self.inference_defaults)
