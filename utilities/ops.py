import tensorflow as tf
import numpy as np

""" Ops from:

https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/ops.py
(the most popular DCGAN implementation on github)
"""

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis.s"""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def weight_variable(shape, name='weight', initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
    return tf.get_variable(name, shape=shape,
        initializer=initializer)

def bias_variable(shape, name='bias'):
    return tf.get_variable(name, shape=shape,
        initializer=tf.constant_initializer(0.0))


def linear(features, n_output, var_scope='linear', no_bias=False,
    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)):
    with tf.variable_scope(var_scope) as scope:
        dim_in = features.get_shape().as_list()[-1]
        weight = weight_variable([dim_in, n_output], name='w', initializer=initializer)

        if no_bias:
            return tf.matmul(features, weight)
        else:
            bias = bias_variable(n_output, name='b')
            return tf.matmul(features, weight) + bias


def conv(features, n_kernel, k_size=4, stride=2, pad='SAME', var_scope='conv'):
    ## Check features is 4D
    with tf.variable_scope(var_scope) as scope:
        dim_in = features.get_shape().as_list()[-1]
        weight = weight_variable([k_size, k_size, dim_in, n_kernel], name='w')
        bias = bias_variable([n_kernel], name='b')

        ## WHYY
        out = tf.nn.conv2d(features, weight, strides=[1, stride, stride, 1], padding=pad)
        oH, oW, oC = out.get_shape().as_list()[1:]
        out = tf.reshape(tf.nn.bias_add(out, bias), [-1, oH, oW, oC])
        return out


def deconv(features, n_kernel, upsample_rate=2, k_size=4, pad='SAME', var_scope='deconv'):
    with tf.variable_scope(var_scope) as scope:
        dim_h_in, dim_w_in, dim_k_in = features.get_shape().as_list()[1:]
        ## output must be whole numbered
        batch_size = tf.shape(features)[0]
        out_h = dim_h_in*upsample_rate
        out_w = dim_w_in*upsample_rate
        output_shape = [batch_size, out_h, out_w, n_kernel]
        # print 'ops/deconv: output_shape', output_shape

        weight = weight_variable([k_size, k_size, n_kernel, dim_k_in], name='w')
        bias = bias_variable([n_kernel], name='b')

        ## why
        out = tf.nn.conv2d_transpose(features, weight, output_shape=output_shape,
            strides=[1, upsample_rate, upsample_rate, 1], padding=pad)
        # out_shape = tf.shape(out)
        out = tf.reshape(tf.nn.bias_add(out, bias), [-1, out_h, out_w, n_kernel])

        return out

def batch_norm(features, training=True, reuse=False, var_scope='batch_norm'):
    out = tf.contrib.layers.batch_norm(features, center=True, scale=True,
        updates_collections=None, is_training=training, reuse=reuse,
        scope=var_scope)

    return out


def lrelu(features, alpha=0.2):
    return tf.maximum(features*alpha, features)
