import tensorflow as tf
import numpy as np
import cv2
import sys, datetime, os, time

sys.path.insert(0, '../..')
import tfmodels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

## ------------------ Hyperparameters --------------------- ##
epochs = 500
iterations = 1000
batch_size = 64
step_start = 0

basedir = 'mnist'
log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
    basedir)
snapshot_path = None

with tf.Session(config=config) as sess:

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)

    dataset = tfmodels.MNISTDataSet(sess=sess,
        batch_size=batch_size,
        capacity=512,
        source_dir='../../assets/mnist_data')
    dataset.print_info()

    print 'test batch:'
    batch_x = next(dataset.iterator)
    print 'batch_x ', batch_x.shape, batch_x.dtype, batch_x.min(), batch_x.max()


    model = tfmodels.VAE(sess=sess,
        batch_size=batch_size,
        dataset=dataset,
        enc_kernels=[64, 128, 512],
        gen_kernels=[128, 64],
        iterator_dataset=True,
        learning_rate=1e-3,
        log_dir=log_dir,
        mode='TRAIN',
        save_dir=save_dir,
        x_dims=[28,28,1],
        z_dim=32)
    model.print_info()

    if snapshot_path is not None:
        model.restore(snapshot_path)

    test_z = np.random.randn(144, model.z_dim)

    for epx in xrange(1, epochs):
        epoch_start = time.time()
        for itx in xrange(iterations):
            model.train_step()

        print 'Epoch [{}] step [{}] time elapsed [{}]s'.format(
            epx, model.global_step, time.time()-epoch_start)
        model.snapshot()

        print 'Sampling from p(x|z), z~N(0,1)'
        outfile = os.path.join(infer_dir, 'step{}.jpg'.format(model.global_step))
        generated_samples = tfmodels.dream_manifold(model, z_manifold_in=test_z)
        cv2.imwrite(outfile, generated_samples)
