# Tensorflow CNN's
This library provides the following interface:
 ```
 import tfmodels
 dataset = tfmodels.ImageMaskDataSet(...)
 model = tfmodels.VGGTraining(datset, ...)
 for ...:
   model.train()

 model.snapshot()

 result = model.inference(images)
 ```
`DataSet` class is a threaded queue dataset that reads continuously from the input directories. `DataSet._preprocessing()` includes basic random coloration, rotation, cropping, and mirroring. To implement a new model, copy+paste an existing model (or use the template), and re-implement the `model()` attribute.

**Note** the default activation (`segmentation/basemodel.py`) is SELU, so the input is scaled to `[-1, 1]`.

## Versioning
```
TensorFlow 1.4
Python 2.7
```

### Bugs
Requesting assistance with the input pipeline:

Best practice for having image-mask pairs, then shuffling them and reading with queues. Temporarily, I saved them as a `(h,w,4)` file where the first 3 channels are the image and the 4th channel is the mask. This is a temporary solution.

#### License
Please provide citation if you use this library for your research.

Copyright 2017 Nathan Ing

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
