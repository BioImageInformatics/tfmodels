from segmentation_basemodel import Segmentation
from regression_basemodel import Regression
from discriminator import SegmentationDiscriminator
from segnet import SegNetTraining, SegNetInference
from vgg import VGGTraining, VGGInference
from fcn8s import FCNTraining, FCNInference
from resnet import ResNetTraining, ResNetInference
from densenet import DenseNetTraining, DenseNetInference
# from resnet_bottleneck import ResNetBottleneckTraining, ResNetBottleneckInference


__all__ = ['Segmentation',
           'Regression',
           'SegmentationDiscriminator',
           'SegNetTraining',
           'SegNetInference',
           'VGGTraining',
           'VGGInference',
           'FCNTraining',
           'FCNInference',
           'ResNetTraining',
           'ResNetInference',
           'DenseNetTraining',
           'DenseNetInference',
       ]
