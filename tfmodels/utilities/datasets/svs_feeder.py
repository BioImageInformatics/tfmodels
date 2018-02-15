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
"""

class SVSFeeder(object):
    svs_feeder_defaults = {
        'svsfile': None,
        'coords': None,
        'level': 0,
    }

    def __init__(self, **kwargs):
        self.svs_feeder_defaults.update(kwargs)
        for key, val in self.svs_feeder_defaults.items():
            setattr(self, key, val)

        assert self.svsffile is not None
        assert self.coords is not None

        self._get_svs_info()


    """ Examine the svs file and populate:
    - dimensions
    - maximum magnification
    - downsample factor from max --> level
    """
    def _get_svs_info(self):
        pass


    def _preprocess(self):
        pass


    def _generator(self):
        pass


    def print_info(self):
        pass
