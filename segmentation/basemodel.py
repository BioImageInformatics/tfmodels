import tensorflow as tf
import numpy as np


class BaseModel(object):
    ## Defaults
    defaults={
        'sess': None,
        'log_dir': None,
        'save_dir': None,
        'training_op_list': [],
        'summary_op_list': [] }

    def __init__(self, **kwargs):
        self.sess = None
        self.log_dir = None
        self.save_dir = None
        self.training_op_list = []
        self.summary_op_list = []
        # self.defaults.update(**kwargs)
        # print self.defaults
        # for key, value in self.defaults.items():
        #     setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.sess is not None

    def print_info(self):
        print '------------------------ BaseModel ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ BaseModel ---------------------- '


    def loss_op(self):
        raise Exception(NotImplementedError)


    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)
