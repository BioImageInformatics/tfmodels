# import sys
# sys.path.insert(0, '.')
# print sys.path

import ops
from discriminator import ConvDiscriminator
from generic import GenericSegmentation
from segnet import SegNetTraining, SegNetInference
from vgg import VGGTraining, VGGInference
from fcn8s import FCNTraining, FCNInference

__all__ = ['ConvDiscriminator',
           'GenericSegmentation',
           'SegNetTraining',
           'SegNetInference',
           'VGGTraining',
           'VGGInference',
           'FCNTraining',
           'FCNInference',
           ]
