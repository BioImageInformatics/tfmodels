import tensorflow as tf
from segmentation_basemodel import Segmentation
from ..utilities.ops import *

class TEMPLATE(Segmentation):
    TEMPLATE_defaults={
        'k_size': None,
        'n_classes': 2,
        'name': 'TEMPLATE',
    }

    def __init__(self, **kwargs):
        self.TEMPLATE_defaults.update(**kwargs)

        assert TEMPLATE_defaults['k_size'] is not None
        assert TEMPLATE_defaults['n_classes'] is not None

        super(TEMPLATE, self).__init__(**self.TEMPLATE_defaults)


    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'TEMPLATE Model'
        k_size = self.k_size
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            y_hat = tf.identity(x_hat)

            return y_hat


class TEMPLATETraining(TEMPLATE):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(TEMPLATETraining, self).__init__(**self.train_defaults)


class TEMPLATEInference(TEMPLATE):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(TEMPLATEInference, self).__init__(**self.inference_defaults)
