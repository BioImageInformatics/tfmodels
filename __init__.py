'''
We want this:
>>> import tfmodels

Then you can do:

dataset = tfmodels.ImageMaskDataSet(...)
model = tfmodels.VGGTraining(dataset, ...)
model.train()
model.snapshot()

later,
inference_model = tfmodels.VGGInference(snapshot_path)
output = inference_model.inference(images)
'''

from utilities import *
from segmentation import *
