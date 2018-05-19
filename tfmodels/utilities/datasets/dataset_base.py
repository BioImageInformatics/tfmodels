"""
Implements a threaded queue for reading images from disk given filenames
Since this is for segmentation, we have to also read masks in the same order

Assume the images and masks are named similarly and are in different folders
"""

from __future__ import print_function

class DataSet(object):
    def __init__(self, **kwargs):
        defaults = {
            'capacity': 5000,
            'name': 'DataSet',
            'seed': 5555,
            'threads': 4,
            'min_holding': 1250,}

        defaults.update(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.batch_size >= 1
        assert self.image_dir is not None


    def print_info(self):
        print('-------------------- {} ---------------------- '.format(self.name))
        for key, value in sorted(self.__dict__.items()):
            print('|\t{}: {}'.format(key, value))
        print('-------------------- {} ---------------------- '.format(self.name))


    def _preprocessing(self, image, mask):
        raise Exception(NotImplementedError)


    def get_batch(self):
        raise Exception(NotImplementedError)
