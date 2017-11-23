import tensorflow as tf
import numpy as np


class BaseGenerator(object):
    generator_defaults = { 'learning_rate': 1e-4, }

    def __init__(self, **kwargs):
        self.generator_defaults.update(**kwargs)
        for key, value in generator_defaults.items():
            setattr(self, key, value)

    def model(self, z_in, keep_prob=0.5, reuse=False, training=True):
        raise Exception(NotImplementedError)
