# import sys
# sys.path.insert(0, '.')
# print sys.path

import ops
from discriminator import ConvDiscriminator
from generic import GenericSegmentation
from segnet import SegNet
from vgg import VGGTraining, VGGInference

__all__ = ['ConvDiscriminator',
           'GenericSegmentation',
           'SegNet',
           'VGGTraining',
           'VGGInference',
           ]
