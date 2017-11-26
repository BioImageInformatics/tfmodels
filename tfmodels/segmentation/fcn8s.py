import tensorflow as tf
from segmentation_basemodel import SegmentationBaseModel

class FCN(SegmentationBaseModel):
    base_defaults={
        'name': 'fcn',
        'snapshot_name': 'fcn'}

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        super(FCN, self).__init__(**self.base_defaults)

        assert self.n_classes is not None
        if self.mode=='TRAIN': assert self.dataset.dstype=='ImageMask'

    ## Layer flow copied from:
    ## https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py
    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'FCN Model'
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=3, stride=1, var_scope='c0_0'))
            c0_1 = nonlin(conv(c0_0, self.conv_kernels[0], k_size=3, stride=1, var_scope='c0_1'))
            c0_pool = tf.nn.max_pool(c0_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c0_pool')
            print '\t c0_pool', c0_pool.get_shape() ## 128

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=3, stride=1, var_scope='c1_0'))
            c1_1 = nonlin(conv(c1_0, self.conv_kernels[1], k_size=3, stride=1, var_scope='c1_1'))
            c1_pool = tf.nn.max_pool(c1_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c1_pool')
            print '\t c1_pool', c1_pool.get_shape() ## 64

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=3, stride=1, var_scope='c2_0'))
            c2_1 = nonlin(conv(c2_0, self.conv_kernels[2], k_size=3, stride=1, var_scope='c2_1'))
            c2_pool = tf.nn.max_pool(c2_1, [1,4,4,1], [1,4,4,1], padding='VALID',
                name='c2_pool')
            print '\t c2_pool', c2_pool.get_shape() ## 32

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=3, stride=1, var_scope='c3_0'))
            c3_0 = tf.contrib.nn.alpha_dropout(c3_0, keep_prob=keep_prob)
            c3_1 = nonlin(conv(c3_0, self.conv_kernels[3], k_size=3, stride=1, var_scope='c3_1'))
            c3_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c3_pool')
            print '\t c3_pool', c3_pool.get_shape()  ## inputs / 16 = 16

            ## The actual ones has one more instead
            # c4_0 = nonlin(conv(c3_pool, self.conv_kernels[4], k_size=3, stride=1, var_scope='c4_0'))
            # c4_0 = tf.contrib.nn.alpha_dropout(c4_0, keep_prob=keep_prob)
            # c4_1 = nonlin(conv(c4_0, self.conv_kernels[4], k_size=3, stride=1, var_scope='c4_1'))
            # c4_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
            #     name='c4_pool')
            # print '\t c4_pool', c4_pool.get_shape()  ## inputs / 32 = 8

            upscore3 = nonlin(deconv(c3_pool, self.n_classes, upsample_rate=16, var_scope='ups3'))
            upscore2 = nonlin(deconv(c2_pool, self.n_classes, upsample_rate=8, var_scope='ups2'))
            upscore1 = nonlin(deconv(c1_pool, self.n_classes, upsample_rate=4, var_scope='ups1'))
            print '\t upscore3', upscore3.get_shape()
            print '\t upscore2', upscore2.get_shape()
            print '\t upscore1', upscore1.get_shape()

            upscore_concat = tf.concat([upscore3, upscore2, upscore1], axis=-1)
            print '\t upscore_concat', upscore_concat.get_shape()
            d0 = nonlin(conv(upscore_concat, deconv_kernels[0], k_size=3, stride=1, var_scope='d0'))
            print '\t d0', d0.get_shape()

            y_hat = deconv(d0, self.n_classes, var_scope='y_hat')
            print '\t y_hat', y_hat.get_shape() ## 128*2 = 256

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
