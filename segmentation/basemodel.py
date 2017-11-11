import tensorflow as tf
import numpy as np


class BaseModel(object):
    ## Defaults
    defaults={
        sess: None,
        log_dir: None,
        save_dir: None,
        training_op_list: [],
        summary_op_list: [] }

    def __init__(self, **kwargs):
        for key, value in defaults.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.sess is not None

    def print_info(self):
        for key, value in self.__dict__:
            print key, value


    def loss_op(self):
        raise Exception(NotImplementedError)


    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)
