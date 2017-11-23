import tensorflow as tf
import numpy as np

class BaseModel(object):
    ## Defaults
    defaults={
        'sess': None,
        'log_dir': None,
        'save_dir': None,
        'name': 'BaseModel',
        'training_op_list': [],
        'summary_op_list': [],
        'snapshot_name': 'snapshot' }

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

        ## Set the nonlinearity for all models
        self.nonlin = tf.nn.selu

    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)

    def get_update_list(self):
        raise Exception(NotImplementedError)

    def summaries(self):
        raise Exception(NotImplementedError)

    def train_step(self, global_step):
        raise Exception(NotImplementedError)

    def snapshot(self, step):
        raise Exception(NotImplementedError)

    def restore(self, snapshot_path):
        raise Exception(NotImplementedError)

    def test_step(self, keep_prob=1.0):
        raise Exception(NotImplementedError)

    def inference(self, x_in, keep_prob):
        raise Exception(NotImplementedError)

    def loss_op(self):
        raise Exception(NotImplementedError)

    def print_info(self):
        print '------------------------ {} ---------------------- '.format(self.name)
        for key, value in sorted(self.__dict__.items()):
            if '_op' in key:
                continue

            if key == 'var_list':
                print '|\t{}:'.format(key)
                for val in value:
                    print '|\t\t{}'.format(val)
                continue

            print '|\t', key, value
        print '------------------------ {} ---------------------- '.format(self.name)

    def _print_info_to_file(self, filename):
        with open(filename, 'w+') as f:
            f.write('---------------------- {} ----------------------\n'.format(self.name))
            for key, value in sorted(self.__dict__.items()):
                if '_op' in key:
                    continue

                if key == 'var_list':
                    f.write('|\t{}:\n'.format(key))
                    for val in value:
                        f.write('|\t\t{}\n'.format(val))
                    continue

                f.write('|\t{}: {}\n'.format(key, value))
            f.write('---------------------- {} ----------------------\n'.format(self.name))
