import tensorflow as tf
import numpy as np
import cv2
import sys, datetime, os, time

sys.path.insert(0, '..')
import tfmodels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

data_home = ''


## ------------------ Hyperparameters --------------------- ##
epochs = 500
iterations = 1000
batch_size = 256
step_start = 0

expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir          = 'vae/logs/{}'.format(expdate)
save_dir         = 'vae/snapshots'
debug_dir        = 'vae/debug'
infer_dir        = 'vae/inference'
snapshot_restore = 'vae/snapshots/vae.ckpt-{}'.format(step_start)

with tf.Session(config=config) as sess:

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)

    dataset = tfmodels.IteratorDataSet(sess=sess,
        batch_size=batch_size,
        capacity=1024,
        # source_dir='/Users/nathaning/Envs/tensorflow/MNIST_data')
        source_dir='/home/nathan/envs/tensorflow/MNIST_data')
    dataset.print_info()

    print 'test batch:'
    batch_x = next(dataset.iterator)
    print 'batch_x ', batch_x.shape, batch_x.dtype, batch_x.min(), batch_x.max()


    model = tfmodels.VAE(sess=sess,
        batch_size=batch_size,
        dataset=dataset,
        enc_kernels=[64, 128, 512],
        gen_kernels=[128, 64],
        global_step= step_start,
        iterator_dataset=True,
        learning_rate=1e-3,
        log_dir=log_dir,
        mode='TRAIN',
        save_dir=save_dir,
        x_dims=[28,28,1],
        z_dim=4)
    model.print_info()
    if step_start > 0:
        model.restore(snapshot_restore)

    test_z = np.random.randn(144, model.z_dim)

    for epx in xrange(1, epochs):
        epoch_start = time.time()
        for itx in xrange(iterations):
            model.train_step()

        print 'Epoch [{}] step [{}] time elapsed [{}]s'.format(
            epx, model.global_step, time.time()-epoch_start)
        model.snapshot()

        print 'Sampling from p(x|z), z~N(0,1)'
        # for zx in xrange(model.z_dim):
        outfile = os.path.join(infer_dir, 'step{}.jpg'.format(model.global_step))
        generated_samples = tfmodels.dream_manifold(model, z_manifold_in=test_z)
        cv2.imwrite(outfile, generated_samples)

    # coord.request_stop()
    # coord.join(threads)
