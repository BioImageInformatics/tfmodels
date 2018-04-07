# Tensorflow CNN's
Library aiming to aid with quick experimentation for new CNN architectures and training schemes using basic TensorFlow.

The focus began as a library for constructing, training, and doing inference with, semantic segmentation CNN's.
In addition to semantic segmentation models, the library also contains base methods for training generative models on image datasets, including Generative Adversarial Networks and Variational Autoencoders.
Additionally, there is some support for image classification tasks.

**Note** instead of using batch norm everywhere, the default activation (see `tfmodels/utilities/basemodel.py`) is SeLU (https://arxiv.org/abs/1706.02515). Accordingly, the inputs should be scaled to `[-1.0, 1.0]` in the dataset loading functions, and models should use `tf.contrib.nn.alpha_dropout` (added as of TensorFlow 1.4.1).

## Versioning
```
Python 2.7.12
TensorFlow 1.5
numpy 1.14
opencv 3.3.0
```

## Getting started
Example scripts for data set interface, training and testing various models are provided under `experiments/`.
- N-class semantic segmentation with various architectures (your data)
- Generative Adversarial Networks (MNIST)
- Variational Autoencoders (MNIST)
- Multi-instance / bagged labels (MNIST)

I've recently revised the way I structure experiments in order to separate my experiments from the structure of this repository.
New examples coming soon ^(TM).

## Modules
- `multi`: multi-instance classification
- `generative`: generative models like VAE's and GAN's
- `segmentation`: special case of fully-supervised conditional generative models.
  - Fully Convolutional Networks (VGG base, 32- 16-, 8- stride versions)
  - SegNet (VGG base with guided upsampling)
  - VGG (without skip connections)
  - Fully Convolutional ResNets
  - Fully Convolutional DenseNets

To create a segmentation network, first define a dataset from a directory on disk (see below), then initialize the model and a training loop.
Each segmentation model comes with two child classes: `*Training` and `*Inference`.
The difference is the required inputs, and training ops are not instantiated when using the `*Inference` versions (check memory savings/initialization times?).
The `experiments` directory contains examples for training some of the models.
Forthcoming will be examples on how to implement a customized network with the `tfmodels` background doing most of the messy work.

For example:

```python
import tfmodels
import tensorflow as tf

## Define training settings
...

## Creates the structure:
##  NAME
##    logs
##    snapshots
##    inference
##    debug
expdirs = tfmodels.make_experiment('NAME', remove_old=False)

dataset = tfmodels.TFRecordImageMask(
  batch_size=batch_size, record_path=data, **kwargs)

with tf.Session() as sess:
  model = tfmodels.DenseNetTraining(
    sess=sess, dataset=dataset, log_dir=expdirs[0],
    save_dir=expdirs[1], **kwargs)

  for iter in xrange(n_iters):
    model.train_step()

    ## print feedback, test, etc.

  ## Save
  model.snapshot()
...
```

- `utilities`: useful classes and functions
  - Test segmentation models
  - Datasets now uses the Dataset API, with utilities for saving and loading images as `tfrecord` format.
    - Basic crop, flipping, and color augmentation included with a future option to add python function preprocessing, e.g. with another Net or through opencv.


### Eager mode
With the new eager execution contrib, we have the option to ditch most of this backend that deals with sessions and storing/calling hooks to ops.
The conveniece may come at a speed penalty, however much I do not yet know.
In any case, it's very nice for example working with mutliple instance learning, where you have some control flow built into the model like for-loops.

There is an example using eager mode for multiple-instance learning on a toy problem.

#### License
Please provide citation if you use this library for your research.

```
PLACEHOLDER BIBTEX
```

Copyright 2017 BioImageInformatics Lab, Cedars-Sinai Medical Center

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
