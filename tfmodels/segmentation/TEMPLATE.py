import tensorflow as tf
from segmentation_basemodel import SegmentationBaseModel
from ..utilities.ops import *

class TEMPLATE(SegmentationBaseModel):
    base_defaults={
        'conv_kernels': None,
        'deconv_kernels': None,
        'k_size': None,
        'name': 'TEMPLATE',
    }

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        super(TEMPLATE, self).__init__(**self.base_defaults)

        assert self.n_classes is not None

    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'TEMPLATE Model'
        k_size = self.k_size
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            y_hat = x_hat

            return y_hat


class TEMPLATETraining(SegNet):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(TEMPLATETraining, self).__init__(**self.train_defaults)


class TEMPLATEInference(SegNet):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(TEMPLATEInference, self).__init__(**self.inference_defaults)
