from segmentation_basemodel import SegmentationBaseModel
from discriminator import SegmentationDiscriminator
# from generic import GenericSegmentation
from segnet import SegNetTraining, SegNetInference
from vgg import VGGTraining, VGGInference
from fcn8s import FCNTraining, FCNInference
from resnet import ResNetTraining, ResNetInference

__all__ = ['SegmentationDiscriminator',
           'SegNetTraining',
           'SegNetInference',
           'VGGTraining',
           'VGGInference',
           'FCNTraining',
           'FCNInference',
           'ResNetTraining',
           'ResNetInference', 
       ]
