import tensorflow as tf
import numpy as np
import datetime, os

"""
BaseModel serves as a template for downstream models:

input -->> tensor operations -->> loss/output

It holds shared operations such as
    - snapshot
    - restore
    - get update list
    - tensorflow boilerplate code
    - printing model settings to terminal, or to file

"""
class BaseModel(object):
    ## Defaults
    base_defaults={
        'sess': None,
        'global_step': 0,
        'log_dir': None,
        'save_dir': None,
        'name': 'base',
        'training_op_list': [],
        'summary_op_list': [],
        }

    def __init__(self, **kwargs):
        self.base_defaults.update(**kwargs)
        for key, value in self.base_defaults.items():
            setattr(self, key, value)

        ## Set nonlinearity for all downstream models
        self.nonlin = tf.nn.selu

    ## In progress for saving each model in its own snapshot (SAVE1)
    # def make_saver(self):
    #     t_vars = tf.trainable_variables()
    #     self.saver = tf.train.Saver([var for var in t_vars if self.name in var.name],
    #         max_to_keep=5,)

    def get_update_list(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if self.name in var.name]

    def inference(self, x_in, keep_prob=1.0):
        raise Exception(NotImplementedError)

    def _loss_op(self):
        raise Exception(NotImplementedError)

    def model(self, x_hat, keep_prob=0.5, reuse=True, training=True):
        raise Exception(NotImplementedError)

    def restore(self, snapshot_path):
        print 'Restoring from {}'.format(snapshot_path)
        try:
            self.saver.restore(self.sess, snapshot_path)
            print 'Success!'
            ## In progress for restoring model + discriminator separately (SAVE1)
            # for saver, snap_path in zip(self.saver):
        except:
            print 'Failed! Continuing without loading snapshot.'

    def snapshot(self):
        ## In progress for saving model + discriminator separately (SAVE1)
        ## have to work up the if/else logic upstream first
        # for saver, snap_dir in zip(self.saver_list, self.snap_dir_list):
        #     print 'Snapshotting to [{}] step [{}]'.format(snap_dir, step),
        #     saver.save(self.sess, snap_dir, global_step=step)

        print 'Snapshotting to [{}] step [{}]'.format(self.snapshot_path, self.global_step),
        self.saver.save(self.sess, self.snapshot_path, global_step=self.global_step)
        print 'Done'

    def summaries(self):
        raise Exception(NotImplementedError)

    def test_step(self, keep_prob=1.0):
        raise Exception(NotImplementedError)

    def _tf_ops(self):
        self.summary_writer = tf.summary.FileWriter(self.log_dir,
            graph=self.sess.graph, flush_secs=30)
        ## Append a model name to the save path
        self.snapshot_path = os.path.join(self.save_dir, '{}.ckpt'.format(self.name))
        # self.make_saver() ## In progress (SAVE1)
        self.saver = tf.train.Saver(max_to_keep=5)

    def train_step(self, global_step):
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
