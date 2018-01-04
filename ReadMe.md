# Tensorflow CNN's
Library aiming to aid with quick experimentation for new CNN architectures and training schemes using basic TensorFlow.

The focus began as a library for constructing, training, and doing inference with, semantic segmentation CNN's.
In addition to semantic segmentation models, the library also contains base methods for training generative models on image datasets, including Generative Adversarial Networks and Variational Autoencoders.
Additionally, there is some support for image classification tasks.

**Note** instead of using batch norm everywhere, the default activation (see `tfmodels/utilities/basemodel.py`) is SeLU. Accordingly, the inputs should be scaled to `[-1.0, 1.0]` in the dataset loading functions, and models should use `tf.contrib.nn.alpha_dropout` (TensorFlow 1.4.1).

## Versioning
```
Python 2.7.12
TensorFlow 1.4.1
numpy 1.13.3
opencv 3.3.0
```

## Getting started
Example scripts for data set interface, training and testing various models are provided under `experiments/`.
- N-class semantic segmentation
- Generative Adversarial Networks (MNIST)
- Variational Autoencoders (MNIST)
- Multi-instance / bagged labels (MNIST)


## Modules
- `multi`: multi-instance classification
- `generative`: generative models like VAE's and GAN's
- `segmentation`: special case of fully-supervised conditional generative models.
  - Fully Convolutional Networks (VGG base)
  - SegNet (VGG base with guided upsampling)
  - VGG (without skip connections)
  - Fully Convolutional ResNets
  - Fully Convolutional DenseNets

To create a segmentation network, first define a dataset from a directory on disk (see below), then initialize the model and a training loop.
Each segmentation model comes with two child classes: `*Training` and `*Inference`.
The difference is the required inputs, and training ops are not instantiated when using the `*Inference` versions (check memory savings/initialization times?).
The `experiments` directory contains examples for training some of the models.

For example:

```
import tfmodels
import tensorflow as tf

## Define training settings, batch_size, data_path, training iterations/epochs, etc.

dataset = tfmodels.ImageComboDataSet(batch_size=batch_size, image_dir=data_path, ...)

with tf.Session() as sess:
  model = tfmodels.DenseNetTraining(sess=sess, dataset=dataset, ...)

  for iter in xrange(n_iters):
    model.train_step()

...
```

- `utilities`: useful classes and functions
  - Test segmentation models
  - Datasets

Datasets, I suspect due to some race conditions when multithreading, are a little special.
In the case we have a fixed number of images, and appropriately matching masks, the `ImageComboDataSet()` class expects a directory with the 3-channel image, and the 1-channel label concatenated together.
Furthermore, it expects all images in this directory to share the same height and width.
To ease this burden, there is a helper function, `tfmodels.write_image_mask_combos()` that will combine two directories of images and labels into the necessary 4-channel combination.


### Bugs
Requesting assistance with the input pipeline:

Best practice for having image-mask pairs, then shuffling them and reading with queues. Temporarily, I saved them as a `(h,w,4)` file where the first 3 channels are the image and the 4th channel is the mask. Then, an `ImageComboDataSet` reads the RGBA-like images, and splits the mask from data as a preprocessing step. This solution is unsatisfactory in the long run.

Planned improvement to move towards the new `Dataset` API in TensorFlow. (https://www.tensorflow.org/programmers_guide/datasets)

#### License
Please provide citation if you use this library for your research.


Copyright 2017 BioImageInformatics Lab, Cedars-Sinai Medical Center

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
