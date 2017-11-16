import tensorflow as tf
import numpy as np
import sys, datetime, os

sys.path.insert(0, '..')
from utilities.datasets import ImageMaskDataSet
from utilities.general import (
    save_image_stack,
    bayesian_inference )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True


data_home = '/Users/nathaning/_original_data/ccRCC_double_stain'
image_dir = '{}/paired_he_ihc_hmm/he'.format(data_home)
mask_dir = '{}/paired_he_ihc_hmm/hmm/4class'.format(data_home)

# data_home = '/home/nathan/histo-seg/semantic-pca/data/_data_origin'
# image_dir = '{}/jpg'.format(data_home)
# mask_dir = '{}/mask'.format(data_home)

batch_size = 64
debug_dir = 'colornorm'

with tf.device('/cpu:0'):
    dataset = ImageMaskDataSet(batch_size=batch_size,
        image_dir=image_dir,
        mask_dir=mask_dir,
        capacity=750,
        min_holding=250,
        threads=4,
        crop_size=512,
        ratio=0.5,
        augmentation='random')
dataset.print_info()

with tf.Session(config=config) as sess:

    print 'Thread coords'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for itx in xrange(100):
        x, y = dataset.get_batch(sess)
        save_image_stack(x[..., ::-1], debug_dir, prefix='img_{}'.format(itx))


    print 'Stopping threads'
    coord.request_stop()
    coord.join(threads)
    print 'Done'
