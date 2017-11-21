import tensorflow as tf
import numpy as np
import sys, datetime, os

sys.path.insert(0, '..')
from segmentation.generic import GenericSegmentation
from segmentation.segnet import SegNetTraining
from utilities.datasets import ImageMaskDataSet
from utilities.general import (
    save_image_stack,
    bayesian_inference )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True

#data_home = '/Users/nathaning/_original_data/ccRCC_double_stain'
#image_dir = '{}/paired_he_ihc_hmm/he'.format(data_home)
#mask_dir = '{}/paired_he_ihc_hmm/hmm/4class'.format(data_home)

# data_home = '/home/nathan/data/ccrcc_tiles'
# image_dir = '{}/he'.format(data_home)
# mask_dir = '{}/hmm/4class'.format(data_home)
data_home = '/home/nathan/histo-seg/semantic-pca/data/_data_origin'
image_dir = '{}/jpg_norm'.format(data_home)
mask_dir = '{}/mask'.format(data_home)

assert os.path.exists(image_dir) and os.path.exists(mask_dir)

## ------------------ Hyperparameters --------------------- ##
epochs = 1000
iterations = 250
batch_size = 16
step_start = 0

expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir = 'pca256segnet/logs/{}'.format(expdate)
save_dir = 'pca256segnet/snapshots'
debug_dir = 'pca256segnet/debug'
snapshot_restore = 'pca256segnet/snapshots/segnet.ckpt-{}'.format(step_start)


with tf.Session(config=config) as sess:
    # with tf.device('/cpu:0'):
    dataset = ImageMaskDataSet(batch_size=batch_size,
        image_dir=image_dir,
        mask_dir=mask_dir,
        capacity=5000,
        min_holding=750,
        threads=2,
        crop_size=512,
        ratio=0.5,
        augmentation='random')
    dataset.print_info()

    model = SegNetTraining(sess=sess,
        dataset=dataset,
        n_classes=4,
        log_dir=log_dir,
        save_dir=save_dir,
        conv_kernels=[64, 64, 64, 64],
        deconv_kernels=[64, 64],
        learning_rate=1e-3,
        x_dims=[256, 256, 3],)
        # adversarial=True, )
        # adversary_lr=5e-7)

    model.print_info()
    if step_start > 0:
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

    save_image_stack(test_x[...,::-1]+1, debug_dir, prefix='x_in_', scale='max', stack_axis=0)
    save_image_stack(test_y, debug_dir, prefix='y__in_', scale=3, stack_axis=0)
    print 'Running initial test'
    for test_idx, test_img in enumerate(test_x_list):
        y_bar_mean, y_bar_var, y_bar = bayesian_inference(model, test_img, 50, keep_prob=0.5)
        # y_bar = model.inference(x_in=test_img, keep_prob=1.0)
        save_image_stack(y_bar, debug_dir,
            prefix='y_bar_{:04d}'.format(test_idx),
            scale=3, ext='png', stack_axis=0)
        save_image_stack(y_bar_mean, debug_dir,
            prefix='y_mean_{:04d}'.format(test_idx),
            scale='max', ext='png', stack_axis=-1)
        save_image_stack(y_bar_var, debug_dir,
            prefix='y_var_{:04d}'.format(test_idx),
            scale='max', ext='png', stack_axis=-1)

    ## --------------------- Optimizing Loop -------------------- ##
    print 'Start'
    global_step = step_start
    for epx in xrange(1, epochs):
        for itx in xrange(iterations):
            global_step += 1
            model.train_step(global_step)

        print 'Epoch [{}] step [{}]'.format(epx, global_step)
        model.snapshot(global_step)

        for test_idx, test_img in enumerate(test_x_list):
            y_bar_mean, y_bar_var, y_bar = bayesian_inference(model, test_img, 50, keep_prob=0.5)
            save_image_stack(y_bar, debug_dir,
                prefix='y_bar_{:04d}'.format(test_idx),
                scale=3, ext='png', stack_axis=0)
            save_image_stack(y_bar_mean, debug_dir,
                prefix='y_mean_{:04d}'.format(test_idx),
                scale='max', ext='png', stack_axis=-1)
            save_image_stack(y_bar_var, debug_dir,
                prefix='y_var_{:04d}'.format(test_idx),
                scale='max', ext='png', stack_axis=-1)


    print 'Stopping threads'
    coord.request_stop()
    coord.join(threads)
    print 'Done'
