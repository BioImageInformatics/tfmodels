# Tensorflow CNN's
This library provides the following interface:
 ```
 import tfmodels
 dataset = tfmodels.ImageMaskDataSet(...)
 model = tfmodels.VGGTraining(datset, ...)
 for _ in ...:
   model.train()

 model.snapshot()

 result = model.inference(images)
 ```
`DataSet` class is a threaded queue dataset that reads continuously from the input directories. `DataSet._preprocessing()` includes basic random coloration, rotation, cropping, and mirroring. To implement a new model, copy+paste an existing model (or use the template), and re-implement the `model()` attribute. See design guidelines below for some more notes.

**Note** the default activation (`segmentation/basemodel.py`) is SELU as of Nov. 20, 2017. Accordingly, the input is scaled to `[-1.0, 1.0]`.

## Versioning
```
TensorFlow 1.4
Python 2.7
```


## Design guidelines
This is how the classes and functions are organized.
1. The DataSet object reads images from a directory on disk. We use a queue to asynchronously pre-process images.
  - DataSets have attributes DataSet.image_op (and optinally DataSet.mask_op) that represent the next image(s) from the file reader.
2. Files under `segmentation/` contain the code for building and training the various models.
  - In `TRAIN` mode, the models require a dataset object to construct themselves.
  - In `TEST` mode we use a `feed_dict` to shove numpy arrays through the network.
  - Model classes have attributes: `training_op_list` and `summary_op_list` which hold the ops for training and dumping to summary. Functions that create ops should end by appending the ops to one of these lists.
  - Models, by default, use their `self.dataset.image_op` as the input to their `model()` method. They use `self.dataset.mask_op` as `y_in` for training.
  - `model()` should return only a tensor of logits the same shape as the input image.
3. Models have options to support Bayesian inference, and adversarial training.
4. `utilities/general.py` has some one-off or randomly helpful functions. Nothing should depend on `utilities/general.py`.


### Bugs
Requesting assistance with the input pipeline:

Best practice for having image-mask pairs, then shuffling them and reading with queues. Temporarily, I saved them as a `(h,w,4)` file where the first 3 channels are the image and the 4th channel is the mask. This solution is unsatisfactory in the long run.

#### License
Please provide citation if you use this library for your research.

Copyright 2017 Nathan Ing

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
