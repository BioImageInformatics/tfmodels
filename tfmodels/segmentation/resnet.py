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
        'modules': 4,
        'name': 'resnet_module',
        'stacks': 5,
    }

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        super(ResNet, self).__init__(**self.base_defaults)

        assert self.n_classes is not None
        ## Check input shape is compatible with the number of downsampling modules
        assert self.modules == len(self.kernels)
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
                print 'Block name', block_name
                signal = self._residual_block(signal, self.kernels[block],
                    block=block, stacks=self.stacks, name_scope='r')
                signal = conv(signal, self.kernels[block+1], stride=2, k_size=1,
                    var_scope=block_name)
                print '\t {}'.format(block_name), signal.get_shape()

            signal = tf.contrib.nn.alpha_dropout(signal, keep_prob=keep_prob)
            signal = self._residual_block(signal, self.kernels[-1], block=self.modules-1,
                stacks=self.stacks, name_scope='r')
            print '\t intermediate: ', signal.get_shape()
            signal = tf.contrib.nn.alpha_dropout(signal, keep_prob=keep_prob)

            for block in xrange(self.modules-1, 1, -1):
                block_name = 'd{}_residual'.format(block)
                signal = self._residual_block(signal, self.kernels[block],
                    block=block, stacks=self.stacks, name_scope='d')
                signal = deconv(signal, self.kernels[block-1], upsample_rate=2, k_size=1,
                    var_scope=block_name)
                print '\t {}'.format(block_name), signal.get_shape()

            signal = self._residual_block(signal, self.kernels[0], block=1,
                stacks=self.stacks, name_scope='d')
            signal = deconv(signal, self.n_classes, upsample_rate=2, k_size=7,
                var_scope='d1_residual')

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
