import tensorflow as tf
import numpy as np
import sys, datetime, os, time

sys.path.insert(0, '..')
import tfmodels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True

#data_home = '/Users/nathaning/_original_data/ccRCC_double_stain'
#image_dir = '{}/paired_he_ihc_hmm/he'.format(data_home)
#mask_dir = '{}/paired_he_ihc_hmm/hmm/4class'.format(data_home)

# data_home = '/home/nathan/histo-seg/semantic-pca/data/train_combo'
# data_home = '/home/chen/env/nathan_tf/data'
record_path = 'gleason_grade.tfrecords'

## ------------------ Hyperparameters --------------------- ##
epochs = 300
batch_size = 32
# iterations = 500/batch_size
iterations = 1000
snapshot_epochs = 10
snapshot_steps = 10000
step_start = 120000

expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir          = 'pca10Xdensenet_20180110/logs/{}'.format(expdate)
save_dir         = 'pca10Xdensenet_20180110/snapshots'
debug_dir        = 'pca10Xdensenet_20180110/debug'
snapshot_restore = 'pca10Xdensenet_20180109/snapshots/densenet.ckpt-{}'.format(step_start)

crop_size = 512
image_ratio = 0.5
prefetch = 1000
threads = 8

with tf.Session(config=config) as sess:
    with tf.device('/cpu:0'):
        # dataset = tfmodels.ImageComboDataSet(batch_size= batch_size,
        #     image_dir= data_home,
        #     image_ext= 'png',
        #     capacity= min_holding + (threads+1)*batch_size,
        #     min_holding= min_holding,
        #     threads= threads,
        #     crop_size= 384*2,
        #     ratio= 0.5,
        #     augmentation= 'random')

        dataset = tfmodlels.TFRecordImageMask(record_path = record_path,
            crop_size = crop_size,
            ratio = image_ratio,
            batch_size = batch_size,
            prefetch = prefetch,
            n_threads = 8,
            sess = sess )
    dataset.print_info()

    # with tf.device('/gpu:0'):
    # model = tfmodels.ResNetTraining(sess=sess,
    model = tfmodels.DenseNetTraining(sess=sess,
        # class_weights=[1.46306, 0.73258, 1.19333, 0.86057],
        class_weights=[1.58918, 0.87155, 1.166423, 0.875136],
        dataset=dataset,
        global_step= step_start,
        k_size= 3,
        dense_stacks= [4, 4, 4, 4, 4],
        growth_rate= 48,
        learning_rate= 1e-8,
        log_dir= log_dir,
        n_classes= 4,
        save_dir= save_dir,
        summarize_grads= True,
        summary_iters= 20,
        summary_image_iters= 250,
        x_dims= [384, 384, 3],)
    model.print_info()

    if step_start > 0:
        model.restore(snapshot_restore)

    ## ------------------- Input Coordinators ------------------- ##
    print 'Starting thread coordinators'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    ## ------------------- Pull a Test batch ------------------- ##
    test_x, test_y = dataset.get_batch(sess)
    test_x_list = np.split(test_x, test_x.shape[0], axis=0)
    test_y_list = np.split(test_y, test_y.shape[0], axis=0)
    print '\t test_x', test_x.shape
    print '\t test_y', test_y.shape

    tfmodels.save_image_stack(test_x[...,::-1]+1, debug_dir,
        prefix='x_in', scale='max', stack_axis=0)
    tfmodels.save_image_stack(test_y, debug_dir,
        prefix='y_in', scale=3, stack_axis=0)
    print 'Running initial test'
    tfmodels.test_bayesian_inference(model, test_x_list, debug_dir)

    ## --------------------- Optimizing Loop -------------------- ##
    print 'Start'

    try:
        if step_start == 0:
            print 'Pretraining'
            model.pretrain()

        print 'Starting at step {}'.format(model.global_step)
        global_step = step_start
        for epx in xrange(1, epochs):
            epoch_start = time.time()
            for itx in xrange(iterations):
                global_step += 1
                model.train_step()

            print 'Epoch [{}] step [{}] time elapsed [{}]s'.format(
                epx, model.global_step, time.time()-epoch_start)

            if model.global_step % snapshot_steps == 0:
                model.snapshot()
                tfmodels.test_bayesian_inference(model, test_x_list, debug_dir)
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
