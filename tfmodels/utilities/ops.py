import tensorflow as tf
import numpy as np

""" Ops from:

https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/ops.py
(the most popular DCGAN implementation on github)
"""

def selu_initializer(shape):
    if len(shape) == 2:
        input_size = shape[0]
    if len(shape) == 4:
        input_size = np.prod(shape[:-1])

    sqrt_1_input = np.sqrt(1.0/input_size)
    print '\t SELU intializer stddev = {:1.5f}'.format(sqrt_1_input)
    return tf.random_normal_initializer(mean=0.0, stddev=sqrt_1_input)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis.s"""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def weight_variable(shape, name='weight',
    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    selu=False):

    if selu:
        initializer = selu_initializer(shape)

    return tf.get_variable(name, shape=shape, initializer=initializer)

def bias_variable(shape, name='bias'):
    return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.0))

def linear(features, n_output, var_scope='linear', no_bias=False,
    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
    selu=False):
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

        print '\t {} dense: {}'.format(var_scope, out.get_shape())
        return out

def conv(features, n_kernel, k_size=4, stride=2, pad='SAME', var_scope='conv', selu=False):
    ## Check features is 4D
    with tf.variable_scope(var_scope) as scope:
        dim_in = features.get_shape().as_list()[-1]
        weight_shape = [k_size, k_size, dim_in, n_kernel]
        weight = weight_variable(weight_shape, name='w', selu=selu)
        bias = bias_variable([n_kernel], name='b')

        ## WHYY
        out = tf.nn.conv2d(features, weight, strides=[1, stride, stride, 1],
            padding=pad)
        oH, oW, oC = out.get_shape().as_list()[1:]
        out = tf.reshape(tf.nn.bias_add(out, bias), [-1, oH, oW, oC])
        print '\t {} conv: {}'.format(var_scope, out.get_shape())
        return out

def deconv(features, n_kernel, upsample_rate=2, k_size=4, pad='SAME',
    var_scope='deconv', selu=False):
    with tf.variable_scope(var_scope) as scope:
        dim_h_in, dim_w_in, dim_k_in = features.get_shape().as_list()[1:]
        ## output must be whole numbered
        batch_size = tf.shape(features)[0]
        out_h = dim_h_in*upsample_rate
        out_w = dim_w_in*upsample_rate
        output_shape = [batch_size, out_h, out_w, n_kernel]
        # print 'ops/deconv: output_shape', output_shape

        weight_shape = [k_size, k_size, n_kernel, dim_k_in]
        weight = weight_variable(weight_shape, name='w', selu=selu)
        bias = bias_variable([n_kernel], name='b')

        ## why
        out = tf.nn.conv2d_transpose(features, weight, output_shape=output_shape,
            strides=[1, upsample_rate, upsample_rate, 1], padding=pad)
        # out_shape = tf.shape(out)
        out = tf.reshape(tf.nn.bias_add(out, bias), [-1, out_h, out_w, n_kernel])
        print '\t {} deconv: {}'.format(var_scope, out.get_shape())
        return out

## BUG batch norm with trainig
def batch_norm(features, reuse=False, training=True, var_scope='batch_norm'):
    with tf.variable_scope(var_scope) as scope:
        ## ReLU
        out = tf.layers.batch_normalization(features, reuse=reuse, training=training)

        return out

def lrelu(features, alpha=0.2):
    return tf.maximum(features*alpha, features)

## https://github.com/tensorflow/tensorflow/issues/2169 // @ThomasWollmann
def unpool(pool, ind, k_size=[1, 2, 2, 1], var_scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:     max pooled output tensor
           ind:      argmax indices
           k_size:   k_size is the same as for the pool
       Return:
           unpool:   unpooling tensor
    """
    with tf.variable_scope(var_scope) as scope:
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * k_size[1], set_input_shape[2] * k_size[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)

        print '\t {} unpool: {}'.format(var_scope, ret.get_shape())
        return ret
#
# def class_weighted_pixelwise_crossentropy(labels, logits, weights=1):
#     sample_weights = tf.reduce_sum(tf.multiply(self.y_in, self.classweights), -1)
#     print '\t segmentation losses sample_weights:', sample_weights
#     xent = tf.losses.softmax_cross_entropy(
#         labels=self.y_in, logits=self.y_hat, weights=sample_weights)
#     print '\t segmentation losses seg_loss:', self.seg_loss
#
#     return xent



## https://github.com/mshunshin/SegNetCMR/blob/master/tfmodel/layers.py
# def unpool_with_argmax(pool, ind, var_scope=None, k_size=[1, 2, 2, 1]):
#     """
#        Unpooling layer after max_pool_with_argmax.
#        Args:
#            pool:   max pooled output tensor
#            ind:      argmax indices
#            ksize:     ksize is the same as for the pool
#        Return:
#            unpool:    unpooling tensor
#     """
#     with tf.variable_scope(var_scope):
#         print 'inputs'
#         print 'pool', pool.get_shape()
#         print 'ind', ind.get_shape()
#         input_shape = pool.get_shape().as_list()
#         batch_size = pool.get_shape().as_list()[0]
#         print 'input_shape', input_shape
#         output_shape = (batch_size, input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3])
#
#         flat_input_size = np.prod(input_shape[1:])
#         print 'flat_input_size', flat_input_size
#         flat_output_shape = [batch_size, output_shape[1] * output_shape[2] * output_shape[3]]
#         print 'flat_output_shape', flat_output_shape
#
#         print 'ind', ind.dtype
#         # output_shape = tf.cast(output_shape, dtype=ind.dtype)
#         # pool_ = tf.reshape(pool, flat_input_size)
#         pool_ = tf.contrib.layers.flatten(pool)
#         print 'pool_', pool_.get_shape()
#         batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), shape=[batch_size, 1,1,1])
#         b = tf.ones_like(ind) * batch_range
#         print 'b', b.get_shape()
#         b = tf.reshape(b, flat_input_size)
#         print 'b', b.get_shape()
#         ind_ = tf.reshape(ind, flat_input_size)
#         print 'ind_', ind_.get_shape()
#         ind_ = tf.concat([b, ind_], 1)
#
#         ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
#         ret = tf.reshape(ret, output_shape)
#         return ret
