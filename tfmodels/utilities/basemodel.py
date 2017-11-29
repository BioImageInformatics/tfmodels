import tensorflow as tf
import numpy as np
import datetime

class BaseModel(object):
    ## Defaults
    base_defaults={
        'sess': None,
        'log_dir': None,
        'save_dir': None,
        'name': 'base',
        'training_op_list': [],
        'summary_op_list': [],
        'snapshot_name': 'snapshot' }

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        for key, value in self.base_defaults.items():
            setattr(self, key, value)

        ## Set nonlinearity for all downstream models
        self.nonlin = tf.nn.selu

    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)

    ## In progress for saving each model in its own snapshot (SAVE1)
    # def make_saver(self):
    #     t_vars = tf.trainable_variables()
    #     self.saver = tf.train.Saver([var for var in t_vars if self.name in var.name],
    #         max_to_keep=5,)

    def get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if self.name in var.name]

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

    def inference(self, x_in, keep_prob=1.0):
        raise Exception(NotImplementedError)

    def loss_op(self):
        raise Exception(NotImplementedError)

    def print_info(self):
        print '------------------------ {} ---------------------- '.format(self.name)
        print '|\t\t TIMESTAMP: {}'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        for key, value in sorted(self.__dict__.items()):
            # if '_op' in key:
            #     continue

            if 'list' in key:
                print '|\t{}:'.format(key)
                for val in value:
                    print '|\t\t{}:'.format(val)
                continue

            print '|\t{}: {}'.format(key, value)
        print '------------------------ {} ---------------------- '.format(self.name)

    def _print_info_to_file(self, filename):
        with open(filename, 'w+') as f:
            f.write('---------------------- {} ----------------------\n'.format(self.name))
            f.write('|\t\t TIMESTAMP: {}\n'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
            for key, value in sorted(self.__dict__.items()):
                # if '_op' in key:
                #     continue

                if 'list' in key:
                    f.write('|\t{}:\n'.format(key))
                    for val in value:
                        f.write('|\t\t{}:\n'.format(val))
                    continue

                f.write('|\t{}: {}\n'.format(key, value))
            f.write('---------------------- {} ----------------------\n'.format(self.name))
