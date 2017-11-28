import tensorflow as tf
import numpy as np
import sys, datetime, os, time

sys.path.insert(0, '..')
import tfmodels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

data_home = ''


## ------------------ Hyperparameters --------------------- ##
epochs = 500
iterations = 100
pretraining = 500
batch_size = 256
step_start = 1

expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir = 'gan/logs/{}'.format(expdate)
save_dir = 'gan/snapshots'
debug_dir = 'gan/debug'
snapshot_restore = 'gan/snapshots/gan.ckpt-{}'.format(step_start)

with tf.Session(config=config) as sess:

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)

    dataset = tfmodels.IteratorDataSet(sess=sess,
        batch_size=batch_size,
        capacity=512,
        # source_dir='/Users/nathaning/Envs/tensorflow/MNIST_data')
        source_dir='/home/nathan/envs/tensorflow/MNIST_data')
    dataset.print_info()

    print 'test batch:'
    batch_x = next(dataset.iterator)
    print 'batch_x ', batch_x.shape, batch_x.dtype, batch_x.min(), batch_x.max()

    model = tfmodels.GAN(sess=sess,
        batch_size=batch_size,
        dataset=dataset,
        dis_kernels=[32, 64, 128],
        gen_kernels=[64, 32],
        iterator_dataset=True,
        log_dir=log_dir,
        mode='TRAIN',
        n_upsamples=2,
        pretraining=pretraining,
        save_dir=save_dir,
        x_dims=[28,28,1],
        z_dim=32)
    model.print_info()

    global_step = step_start
    for epx in xrange(1, epochs):
        epoch_start = time.time()
        for itx in xrange(iterations):
            global_step += 1
            model.train_step(global_step)

        print 'Epoch [{}] step [{}] time elapsed [{}]s'.format(
            epx, global_step, time.time()-epoch_start)
        # model.snapshot(global_step)


    # coord.request_stop()
    # coord.join(threads)
