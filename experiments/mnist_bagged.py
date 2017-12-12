import tensorflow as tf
import numpy as np
import sys, datetime, os, time

from tensorflow.examples.tutorials.mnist import input_data

sys.path.insert(0, '..')
import tfmodels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True

mnist_data_path = '/Users/nathaning/Envs/tensorflow/MNIST_data'
mnist_data = input_data.read_data_sets(mnist_data_path)

"""
Bagged MNIST is a toy dataset using the MNIST digits data.
We have images of digits: {0,1,2,3,4,5,6,7,8,9}
First we choose one, or some combination, to be the "positive" class.

For training we draw **sets** of digits. Each set is labelled "positive" if
it contains a "positive" class element.

e.g. if positive = 0
x = [1,2,3,5,2,3,0], y = 1
x = [1,2,3,5,2,3,9], y = 0

Training proceeds to predict positive bags.

What we recover is a classifier that maximizes the expected value of

p(y=1 | x=positive) **(prove it)

without explicitly stating which element is the "positive" one.
"""
## ------------------ Hyperparameters --------------------- ##
epochs = 200
iterations = 500
snapshot_epochs = 5
step_start = 0

batch_size = 128
samples = 20

expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir          = 'bagged/logs/{}'.format(expdate)
save_dir         = 'bagged/snapshots'
debug_dir        = 'bagged/debug'
snapshot_restore = 'bagged/snapshots/resnet.ckpt-{}'.format(step_start)

training_dataset = tfmodels.BaggedMNIST(
    batch_size     = batch_size,
    samples        = samples,
    positive_class = 1,
    data           = mnist_data.train,
    mode           = 'Train'
    )
testing_dataset = tfmodels.BaggedMNIST(
    batch_size     = batch_size,
    samples        = samples,
    positive_class = 1,
    data           = mnist_data.test,
    mode           = 'Test'
    )

with tf.Session(config=config) as sess:
    model = tfmodels.ImageBagModel(
        dataset         = training_dataset,
        log_dir         = log_dir,
        save_dir        = save_dir,
        sess            = sess,
        # summarize_grads = True,
        # summarize_vars  = True,
        )
    model.print_info()

    print 'Starting training'
    for epoch in xrange(1, epochs):
        for _ in xrange(iterations):
            model.train_step()

        ## Test bags

        ## Test encoder network to discriminate individual examples
        print 'Testing individual obs:',
        test_x, test_y = testing_dataset.normal_batch(batch_size=128)
        test_y_hat = sess.run(model.z_individual, feed_dict={
            model.x_individual: test_x })
        print 'test_y_hat:', test_y_hat.shape
        test_accuracy = np.mean(np.argmax(test_y,axis=1) == np.argmax(test_y_hat,axis=1))

        print 'Epoch [{:05d}]; x_i acc: [{:03.2f}]'.format(epoch, test_accuracy)
