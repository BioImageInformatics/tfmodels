import tensorflow as tf
import numpy as np
import sys, datetime

sys.path.insert(0, '..')
from segmentation.generic import GenericSegmentation
from utilities.datasets import ImageMaskDataSet
from utilities.general import save_image_stack

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#data_home = '/Users/nathaning/_original_data/ccRCC_double_stain'
#image_dir = '{}/paired_he_ihc_hmm/he'.format(data_home)
#mask_dir = '{}/paired_he_ihc_hmm/hmm/4class'.format(data_home)

data_home = '/home/nathan/data/ccrcc_tiles'
image_dir = '{}/he'.format(data_home)
mask_dir = '{}/hmm/4class'.format(data_home)

dataset = ImageMaskDataSet(batch_size=64,
    image_dir=image_dir,
    mask_dir=mask_dir,
    capacity=1500,
    min_holding=500,
    threads=8,
    crop_size=512,
    ratio=0.5)
dataset.print_info()

expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir = 'ccrcc/logs/{}'.format(expdate)
save_dir = 'ccrcc/snapshots'
debug_dir = 'ccrcc/debug'


epochs = 100
iterations = 1000

with tf.Session(config=config) as sess:
    model = GenericSegmentation(sess=sess,
        dataset=dataset,
        n_classes=4,
        log_dir=log_dir,
        save_dir=save_dir,
        learning_rate=1e-3,)
        #adversarial=True)
    model.print_info()

    ## ------------------- Input Coordinators ------------------- ##
    print 'Thread coordinators'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # x_in, y_in = dataset.get_batch(sess)

    ## ------------------- Preliminary Train ------------------- ##
    # print 'Testing'
    # x_in, y_in, y_hat = model.test_step()
    # print '\t x_in', x_in.shape, x_in.dtype, x_in.min(), x_in.max()
    # print '\t y_in', y_in.shape, y_in.dtype, y_in.min(), y_in.max(), np.unique(y_in)
    # print '\t y_hat', y_hat.shape, y_hat.dtype, y_hat.min(), y_hat.max(), np.unique(y_hat)
    #
    # save_image_stack(x_in[:,:,:,::-1], debug_dir, prefix='x_in_0')
    # save_image_stack(y_in, debug_dir, prefix='y_in_0', onehot=True, scale=3)
    # save_image_stack(y_hat, debug_dir, prefix='y_hat_0', onehot=True, scale=3)

    ## --------------------- Optimizing Loop -------------------- ##
    print 'Start'
    global_step = 0
    for epx in xrange(1, epochs):
        for itx in xrange(iterations):
            global_step += 1
            model.train_step(global_step)

        print 'Epoch [{}] step [{}]'.format(epx, global_step)
        x_in, y_in, y_hat = model.test_step()
        # print '\t x_in', x_in.shape, x_in.dtype, x_in.min(), x_in.max()
        # print '\t y_in', y_in.shape, y_in.dtype, y_in.min(), y_in.max(), np.unique(y_in)
        # print '\t y_hat', y_hat.shape, y_hat.dtype, y_hat.min(), y_hat.max(), np.unique(y_hat)

        # save_image_stack(x_in[:,:,:,::-1], debug_dir, prefix='x_in_{}'.format(epx))
        # save_image_stack(y_in, debug_dir, prefix='y_in_{}'.format(epx), onehot=True, scale=3)
        # save_image_stack(y_hat, debug_dir, prefix='y_hat_{}'.format(epx), onehot=True, scale=3)

        save_image_stack(x_in[:,:,:,::-1], debug_dir, prefix='x_in')
        save_image_stack(y_in, debug_dir, prefix='y_in', onehot=True, scale=3)
        save_image_stack(y_hat, debug_dir, prefix='y_hat', onehot=True, scale=3)

    coord.request_stop()
    coord.join(threads)
