from __future__ import print_function
import tensorflow as tf
import numpy as np

""" Ops based on:
https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/ops.py
"""
def selu_initializer(shape):
    if len(shape) == 2:
        input_size = shape[0]
    if len(shape) == 4:
        input_size = np.prod(shape[:-1])
    sqrt_1_input = np.sqrt(1.0/input_size)
    # print('\t\t SELU intializer stddev = {:1.5f}'.format(sqrt_1_input))
    return tf.random_normal_initializer(mean=0.0, stddev=sqrt_1_input)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis.s"""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def crop_concat(x, y):
    """ Crop x to match y on dim 1 & 2 then return the concatenated tensors

    https://github.com/jakeret/tf_unet/blob/master/tf_unet/layers.py
    """
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    x_shape_ = x.get_shape()
    y_shape_ = y.get_shape()
    expected_shape = [-1, y_shape_[1], y_shape_[2], x_shape_[-1]+y_shape_[-1]]

    offsets = [0, (x_shape[1] - y_shape[1]) // 2, (x_shape[2] - y_shape[2]) // 2, 0]
    size = [-1, y_shape[1], y_shape[2], -1]
    # size = tf.constant([-1, y_shape[1], y_shape[2], x_shape[-1]])
    x_crop = tf.slice(x, offsets, size)
    x_y = tf.concat([x_crop, y], -1)
    x_y = tf.reshape(x_y, expected_shape)
    return x_y


def weight_variable(shape, name='weight',
    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    # initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
    selu=True):
    if selu:
        initializer = selu_initializer(shape)
    return tf.get_variable(name, shape=shape, initializer=initializer)

def bias_variable(shape, name='bias'):
    return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.0))

def linear(features, n_output, var_scope='linear', no_bias=False,
    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
    selu=True):
    with tf.variable_scope(var_scope) as scope:
        dim_in = features.get_shape().as_list()[-1]
        weight_shape = [dim_in, n_output]
        weight = weight_variable(weight_shape, name='w',
            initializer=initializer, selu=selu)
        if no_bias:
            out = tf.matmul(features, weight)
        else:
            bias = bias_variable(n_output, name='b')
            out = tf.matmul(features, weight) + bias
        # print('\t {} dense: {} --> {}'.format(var_scope, features.get_shape(), out.get_shape()))
        return out

def conv(features, n_kernel, k_size=4, stride=2, pad='SAME', var_scope='conv',
    dilation=None, selu=True, no_bias=False):
    ## Check features is 4D
    with tf.variable_scope(var_scope) as scope:
        dim_in = features.get_shape().as_list()[-1]
        weight_shape = [k_size, k_size, dim_in, n_kernel]
        weight = weight_variable(weight_shape, name='w', selu=selu)
        ## WHYY
        if dilation is not None:
            assert stride==1
            # dilation = 1
            # print('\t using dilation, {}'.format(dilation))
            dilation = [dilation, dilation]
        # else:
        #     print('\t using no dilation')

        out = tf.nn.convolution(features, weight, strides=[stride, stride],
            padding=pad, dilation_rate=dilation)
        # if dilation is not None:
        #     out = tf.nn.atrous_conv2d(features, weight, strides=[1, stride, stride, 1],
        #         padding=pad)
        # else:
        #     out = tf.nn.conv2d(features, weight, strides=[1, stride, stride, 1],
        #         padding=pad)
        if no_bias:
            pass
        else:
            oH, oW, oC = out.get_shape().as_list()[1:]
            bias = bias_variable([n_kernel], name='b')
            out = tf.reshape(tf.nn.bias_add(out, bias), [-1, oH, oW, oC])
        # print('\t {} conv: {} --> {}'.format(var_scope, features.get_shape(), out.get_shape()))
        return out

def deconv(features, n_kernel, upsample_rate=2, k_size=4, pad='SAME', var_scope='deconv',
    dilation=None, selu=True, no_bias=False):
    with tf.variable_scope(var_scope) as scope:
        dim_h_in, dim_w_in, dim_k_in = features.get_shape().as_list()[1:]
        ## output must be whole numbered
        batch_size = tf.shape(features)[0]
        out_h = dim_h_in*upsample_rate
        out_w = dim_w_in*upsample_rate
        output_shape = [batch_size, out_h, out_w, n_kernel]
        weight_shape = [k_size, k_size, n_kernel, dim_k_in]
        weight = weight_variable(weight_shape, name='w', selu=selu)
        ## why
        out = tf.nn.conv2d_transpose(features, weight, output_shape=output_shape,
            strides=[1, upsample_rate, upsample_rate, 1], padding=pad)
        if no_bias:
            pass
        else:
            bias = bias_variable([n_kernel], name='b')
            out = tf.reshape(tf.nn.bias_add(out, bias), [-1, out_h, out_w, n_kernel])
        # print('\t {} deconv: {} --> {}'.format(var_scope, features.get_shape(), out.get_shape()))
        return out

## BUG batch norm with training
def batch_norm(features, momentum=0.9, reuse=False, training=True, var_scope='batch_norm'):
    # print('Batch norm input shape: {}'.format(features.get_shape()))
    with tf.variable_scope(var_scope) as scope:
        ## ReLU
        out = tf.layers.batch_normalization(features, renorm=True, reuse=reuse,
            momentum=momentum,
            fused=None,
            training=training)
        return out

def lrelu(features, alpha=0.2):
    return tf.maximum(features*alpha, features)

## https://github.com/tensorflow/tensorflow/issues/2169 // @ThomasWollmann
## I find it functional, but feels sluggish
def unpool(features, ind, k_size=[1, 2, 2, 1], var_scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           features:     max pooled output tensor
           ind:      argmax indices
           k_size:   k_size is the same as for the features
       Return:
           unpool:   unpooling tensor
    """
    with tf.variable_scope(var_scope) as scope:
        input_shape = tf.shape(features)
        output_shape = [input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(features, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        out = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        out = tf.reshape(out, output_shape)

        set_input_shape = features.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * k_size[1], set_input_shape[2] * k_size[2], set_input_shape[3]]
        out.set_shape(set_output_shape)

        # print('\t {} unpool: {} --> {}'.format(var_scope, features.get_shape(), out.get_shape()))
        return out
