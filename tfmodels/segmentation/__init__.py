from segmentation_basemodel import SegmentationBaseModel
from discriminator import SegmentationDiscriminator
from segnet import SegNetTraining, SegNetInference
from vgg import VGGTraining, VGGInference
from fcn8s import FCNTraining, FCNInference
from resnet import ResNetTraining, ResNetInference
# from resnet_bottleneck import ResNetBottleneckTraining, ResNetBottleneckInference
from resnet_module import ResNetModuleTraining, ResNetModuleInference


__all__ = ['SegmentationDiscriminator',
           'SegNetTraining',
           'SegNetInference',
           'VGGTraining',
           'VGGInference',
           'FCNTraining',
           'FCNInference',
           'ResNetTraining',
           'ResNetInference',
           'ResNetModuleTraining',
           'ResNetModuleInference',
       ]
