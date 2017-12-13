## Demo scripts

These scripts demonstrate how to perform model training


### GAN


### VAE


### Multi-instance
Multiple-instance, or bagged label, problems come up quite often.
The script `mnist_bagged.py` trains a classifier on the bagged MNIST toy dataset.
We pick a "positive" class (or classes), and train on sets of images that either do or do not contain one or more positive examples.

For instance, we choose [0,1,2] to be positive:


**Positive bag** (contains 0,1,2) | **Negative bag** (no 0,1 or 2)
:--: | :--:
<img src="../assets/img_6_1.jpg" width=""> | <img src="../assets/img_7_0.jpg" width="">

Setting the encoder function to return a binary positive / negative label results in a classifier that determines individual instances, x_i, belonging to the positive set. After only 500 iterations of training batch size 128, and a simple feed-forward model we get good results:

**Classified Positive** | **Classified Negative**
:--: | :--:
<img src="../assets/pos_002.jpg" width="48"> | <img src="../assets/neg_027.jpg" width="48">
<img src="../assets/pos_004.jpg" width="48"> | <img src="../assets/neg_028.jpg" width="48">
<img src="../assets/pos_005.jpg" width="48"> | <img src="../assets/neg_029.jpg" width="48">
<img src="../assets/pos_006.jpg" width="48"> | <img src="../assets/neg_035.jpg" width="48">
<img src="../assets/pos_008.jpg" width="48"> | <img src="../assets/neg_036.jpg" width="48">

<!-- If [3,6] are chosen positive, we see more mixups in the individual classifier, while bagged examples remain high accuracy.

**Classified Positive** | **Classified Negative**
:--: | :--:
<img src="../assets/6_3/pos_082.jpg" width="64"> | <img src="../assets/6_3/neg_117.jpg" width="64">
<img src="../assets/6_3/pos_087.jpg" width="64"> | <img src="../assets/6_3/neg_112.jpg" width="64">
<img src="../assets/6_3/pos_104.jpg" width="64"> | <img src="../assets/6_3/neg_108.jpg" width="64">
<img src="../assets/6_3/pos_106.jpg" width="64"> | <img src="../assets/6_3/neg_109.jpg" width="64">
<img src="../assets/6_3/pos_110.jpg" width="64"> | <img src="../assets/6_3/neg_039.jpg" width="64"> -->
