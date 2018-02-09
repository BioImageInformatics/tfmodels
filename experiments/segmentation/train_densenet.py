import tensorflow as tf
import numpy as np
import sys, datetime, os, time

sys.path.insert(0, '../..')
import tfmodels
sys.path.insert(0, '.')
from model import Training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
training_record = 'gleason_grade_train.tfrecord'
testing_record = 'gleason_grade_test.tfrecord'

## ------------------ Hyperparameters --------------------- ##
epochs = 300
batch_size = 32
iterations = 1000
snapshot_epochs = 10

basedir = 'gleason'
log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
    basedir)
snapshot_path = ''

crop_size = 512
image_ratio = 0.5
prefetch = 1000
threads = 8

lr_start = 1e-4
lr_gamma = 1e-4
lr_step = 10000
def learning_rate(step):
    return lr_start * np.exp(-lr_gamma * step)


with tf.Session(config=config) as sess:
    dataset = tfmodlels.TFRecordImageMask(training_record = training_record,
        testing_record = testing_record,
        crop_size = crop_size,
        ratio = image_ratio,
        batch_size = batch_size,
        prefetch = prefetch,
        n_threads = 8,
        n_classes = 2,
        sess = sess )
    dataset.print_info()

    # with tf.device('/gpu:0'):
    # model = tfmodels.ResNetTraining(sess=sess,
    model = tfmodels.DenseNetTraining(sess=sess,
        dataset=dataset,
        k_size= 3,
        dense_stacks= [4, 6, 8, 8],
        growth_rate= 32,
        learning_rate= 1e-8,
        log_dir= log_dir,
        n_classes= 4,
        save_dir= save_dir,
        summarize_grads= True,
        summary_iters= 20,
        summary_image_iters= 250,
        x_dims= [256, 256, 3],)
    model.print_info()

    if snapshot_path > 0:
        model.restore(snapshot_path)

    ## ------------------- Input Coordinators ------------------- ##
    print 'Starting thread coordinators'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    ## --------------------- Optimizing Loop -------------------- ##
    print 'Start'

    try:
        global_step = 0
        for epx in xrange(1, epochs):
            epoch_start = time.time()
            for itx in xrange(iterations):
                global_step += 1
                model.train_step()

            print 'Epoch [{}] step [{}] time elapsed [{}]s'.format(
                epx, model.global_step, time.time()-epoch_start)

            ## Run a test    
            model.test()

            if epx % snapshot_epochs == 0:
                model.snapshot()

    except Exception as e:
        print 'Caught exception'
        print e.__doc__
        print e.message
    finally:
        model.snapshot()
        print 'Stopping threads'
        coord.request_stop()
        coord.join(threads)
        print 'Done'
