import tensorflow as tf
import numpy as np
import sys, datetime, os

sys.path.insert(0, '..')
from segmentation.generic import GenericSegmentation
from utilities.datasets import ImageMaskDataSet
from utilities.general import (
    save_image_stack,
    bayesian_inference )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

data_home = '/Users/nathaning/_original_data/ccRCC_double_stain'
image_dir = '{}/paired_he_ihc_hmm/he'.format(data_home)
mask_dir = '{}/paired_he_ihc_hmm/hmm/4class'.format(data_home)

# data_home = '/home/nathan/data/ccrcc_tiles'
# image_dir = '{}/he'.format(data_home)
# mask_dir = '{}/hmm/4class'.format(data_home)

assert os.path.exists(image_dir) and os.path.exists(mask_dir)

dataset = ImageMaskDataSet(batch_size=16,
    image_dir=image_dir,
    mask_dir=mask_dir,
    capacity=500,
    min_holding=100,
    threads=8,
    crop_size=512,
    ratio=0.5)
dataset.print_info()

expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir = 'bayes/logs/{}'.format(expdate)
save_dir = 'bayes/snapshots'
debug_dir = 'bayes/debug'

snapshot_restore = 'bayes/snapshots/generic_segmentation.ckpt-25'

epochs = 100
iterations = 25

with tf.Session(config=config) as sess:
    model = GenericSegmentation(sess=sess,
        dataset=dataset,
        n_classes=4,
        log_dir=log_dir,
        save_dir=save_dir,
        learning_rate=1e-3,
        x_dims=[256, 256, 3])
        #adversarial=True)
    model.print_info()
    model.restore(snapshot_restore)

    ## ------------------- Input Coordinators ------------------- ##
    print 'Thread coordinators'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    ## ------------------- Pull a Test batch ------------------- ##
    test_x, test_y = dataset.get_batch(sess)
    test_x_list = np.split(test_x, test_x.shape[0], axis=0)
    test_y_list = np.split(test_y, test_y.shape[0], axis=0)
    print '\t test_x', test_x.shape
    print '\t test_y', test_y.shape

    print 'Running initial test'
    for test_idx, test_img in enumerate(test_x_list):
        y_bar_mean, y_bar_var = bayesian_inference(model, test_img, 25)
        save_image_stack(y_bar_mean, debug_dir,
            prefix='y_mean_{:04d}'.format(test_idx),
            scale='max', stack_axis=-1)
        save_image_stack(y_bar_var, debug_dir,
            prefix='y_var_{:04d}'.format(test_idx),
            scale='max', stack_axis=-1)

    ## --------------------- Optimizing Loop -------------------- ##
    print 'Start'
    global_step = 0
    for epx in xrange(1, epochs):
        for itx in xrange(iterations):
            global_step += 1
            model.train_step(global_step)

        print 'Epoch [{}] step [{}]'.format(epx, global_step)
        model.snapshot(global_step)

        for test_idx, test_img in enumerate(test_x_list):
            y_bar_mean, y_bar_var = bayesian_inference(model, test_img, 25)
            save_image_stack(y_bar_mean, debug_dir,
                prefix='y_mean_{:04d}'.format(test_idx),
                scale='max', stack_axis=-1)
            save_image_stack(y_bar_var, debug_dir,
                prefix='y_var_{:04d}'.format(test_idx),
                scale='max', stack_axis=-1)



    coord.request_stop()
    coord.join(threads)
