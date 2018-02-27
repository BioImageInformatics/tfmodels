import tensorflow as tf
import cv2
import numpy as np

from opeslide import OpenSlide

"""
SVS loader to asynchronously pull from coordinates in an svs
and feed an image_op.

Accept a single svs file, and a list of coordinates and the level to read from
use py_func option in tf.data.Dataset

use py_func to preprocess. Accept a function to use for preprocessing
from the outside.
function should be of the form f(mask) --> mask_ , where dim(mask) = dim(mask_)

https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator

"""

class SVSFeeder(object):
    svs_feeder_defaults = {
        'svsfile': None,
        'tile_size': 512,
        'downsample': 0.5,
        'overlap_fact': 1.5,
        'level': 0, }

    def __init__(self, **kwargs):
        self.svs_feeder_defaults.update(kwargs)
        for key, val in self.svs_feeder_defaults.items():
            setattr(self, key, val)

        assert self.svsffile is not None
        assert self.coords is not None

        self._prepare_generator()

        self.dataset = tf.data.Dataset.from_generator(self._generator,
            tf.float32, tf.TensorShape([]))
        self.iterator = self.dataset.make_initializable_iterator()
        self.image_op = self.iterator.get_next()


    """ Examine the svs file and populate:
    - dimensions
    - maximum magnification
    - downsample factor from max --> level
    """
    def _get_svs_info(self):
        pass


    """ Return list of foreground coordinates
    """
    def _find_foreground(self):
        pass


    """ Function to be run before constructing the dataset """
    def _prepare_generator(self):
        pass


    def _preprocess(self):
        pass


    """ Create a dataset that knows what coordinates to feed
    it should exhaust itself after one time through the coordinates
    """
    def _generator(self):
        pass


    def print_info(self):
        pass
