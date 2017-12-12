# Tensorflow CNN's
This library contains base methods for training segmentation (generally, image-to-image) CNN's and provides the following interface:
 ```
 import tfmodels
 dataset = tfmodels.ImageMaskDataSet(...)
 model = tfmodels.VGGTraining(datset, ...)
 for _ in ...:
   model.train()

 model.snapshot()

 result = model.inference(images)
 ```

To implement a new model, copy-paste one of the existing models and re-implement the `model()` method.

The libaray also contains base methods for training generative models including Generative Adverarial Networks and Variational Autoencoders.

**Note** the default activation (set in `tfmodels/utilities/basemodel.py`) is SeLU. Accordingly, the input is scaled to `[-1.0, 1.0]` in the dataset loading functions, and we should use `tf.contrib.nn.alpha_dropout`.


## Versioning
```
Python 2.7
TensorFlow 1.4
numpy
cv2
```

## Getting started
Example scripts for dataset interface, training and testing various models are provided under `experiments/`.


## Modules
- `multi`: multi-instance classification and applications
- `generative`: generative models like VAE's and GAN's
- `segmentation`: special case of conditional generative models.
- `utilities`: useful classes and functions


### Bugs
Requesting assistance with the input pipeline:

Best practice for having image-mask pairs, then shuffling them and reading with queues. Temporarily, I saved them as a `(h,w,4)` file where the first 3 channels are the image and the 4th channel is the mask. Then, an `ImageComboDataSet` reads the RGBA-like images, and splits the mask from data as a preprocessing step. This solution is unsatisfactory in the long run.

#### License
Please provide citation if you use this library for your research.

<!-- Copyright 2017 Nathan Ing

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. -->
