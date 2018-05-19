from __future__ import print_function
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

## https://stackoverflow.com/questions/29831489/numpy-1-hot-array
def np_onehot(a, depth):
    b = np.zeros((a.shape[0], depth))
    b[np.arange(a.shape[0]), a] = 1
    return b

"""
Collect examples of positive class, and all others
Support a list of positive labels
"""
def _collect_mnist(mnist, positive_class=0):
    if not isinstance(positive_class, (list, tuple)):
        positive_class = [positive_class]
    ## Gather x
    positive_x = []
    negative_x = []
    for y in range(0, 10):
        if y in positive_class:
            positive_x.append(mnist.images[mnist.labels==y, :])
        else:
            negative_x.append(mnist.images[mnist.labels==y, :])
        # x_out = mnist.images[mnist.labels==y, :]
        # # mnist_out[y] = np.vstack([train_x, test_x])
        # mnist_out[y] = x_out

    # positive_x = mnist_out[positive_class]
    # negative_x = [mnist_out[c] for c in range(10) if c != positive_class]
    positive_x = np.vstack(positive_x)
    negative_x = np.vstack(negative_x)

    # np.random.shuffle(positive_x)
    # np.random.shuffle(negative_x)

    return negative_x, positive_x

"""
Produces batches like:
x = [batch_size, samples, [dimensions]], y = [batch_size]

where y is 0 or 1 for a batch containing the positive class

batches containing the positive class may have a variable number of
positive examples
"""
class BaggedMNIST(object):
    def __init__(self, **kwargs):
        bag_mnist_defaults = {
            'as_images': False,
            'batch_size': 64,
            'name': 'MNISTDataSet',
            'onehot': True,
            'positive_class': [0],
            'positive_freq': 0.5,
            'positive_rate': 0.1, ## Rate of positive class in each positive bag
            'samples': 10,
            'data': None, ## One of mnist.train or mnist.test
            'mode': 'Train',
            # 'source_dir': None,
        }

        print('Initializing Bagged MNIST dataset')
        bag_mnist_defaults.update(**kwargs)
        for key, value in bag_mnist_defaults.items():
            setattr(self, key, value)

        ## load without onehot for ez
        # self.data = input_data.read_data_sets(self.source_dir)

        print('Using positive class:', self.positive_class)
        self.negative_x, self.positive_x = _collect_mnist(self.data,
            positive_class=self.positive_class)
        self._count_examples()
        self.iterator = self.iterator_fn()


    def _count_examples(self):
        self.negative_count = self.negative_x.shape[0]
        self.positive_count = self.positive_x.shape[0]

        print('Got negative examples', self.negative_count)
        print('Got positive examples', self.positive_count)


    def _get_x(self, y):
        ## positive
        if y:
            ## portion of samples to be positive:
            portion = np.random.binomial(self.samples-1, self.positive_rate)
            negative_portion = self.samples-portion
            positive_x = self.positive_x[np.random.choice(self.positive_count, portion), :]
            filler_x = self.negative_x[np.random.choice(self.negative_count, negative_portion), :]

            batch_x = np.vstack([positive_x, filler_x])
            np.random.shuffle(batch_x)
        if not y:
            indices = np.random.choice(self.negative_count, self.samples)
            batch_x = self.negative_x[indices, :]

        if self.as_images:
            batch_x = [x.reshape(28, 28) for x in batch_x]
            batch_x = [np.expand_dims(x, 0) for x in batch_x]
            batch_x = [np.expand_dims(x, -1) for x in batch_x]
            batch_x = np.concatenate(batch_x, 0)

        return np.expand_dims(batch_x, 0)


    def iterator_fn(self):
        while True:
            batch_y = np.random.binomial(1, self.positive_freq, self.batch_size)
            batch_x = [self._get_x(y) for y in batch_y]
            batch_x = np.concatenate(batch_x, 0)

            ## Move to [-1, 1] for SELU
            # batch_x = batch_x * (2) - 1

            if self.onehot:
                batch_y = np_onehot(batch_y, 2)

            yield batch_x, batch_y


    def choice_positive(self, n=1):
        return np.random.choice(range(self.positive_count), n)

    def choice_negative(self, n=1):
        return np.random.choice(range(self.negative_count), n)


    ## Return a batch labelled positive-non-positive
    def normal_batch(self, batch_size):
        batch_y = np.random.binomial(1, 0.5, batch_size)
        batch_x = []
        for y in batch_y:
            if y:
                idx = self.choice_positive()
                batch_x.append(self.positive_x[idx, :])
            else:
                idx = self.choice_negative()
                batch_x.append(self.negative_x[idx, :])

        if self.as_images:
            batch_x = [x.reshape(28, 28) for x in batch_x]
            batch_x = [np.expand_dims(x, 0) for x in batch_x]
            batch_x = [np.expand_dims(x, -1) for x in batch_x]
            batch_x = np.concatenate(batch_x, 0)
        else:
            batch_x = np.concatenate(batch_x, 0)

        # batch_x = batch_x * 2 - 1

        if self.onehot:
            batch_y = np_onehot(batch_y, 2)

        return batch_x, batch_y


    def print_info(self):
        print('---------------------- {} ---------------------- '.format(self.name))
        for key, value in sorted(self.__dict__.items()):
            print('|\t', key, value)
        print('---------------------- {} ---------------------- '.format(self.name))



"""
Using TF's built in MNIST dataset

https://stackoverflow.com/questions/43231958/filling-queue-from-python-iterator
"""
class MNISTDataSet(object):
    mnist_dataset_defaults = {
        'batch_size': 64,
        'mode': 'TRAIN',
        'name': 'MNISTDataSet',
        'source_dir': None,
    }

    def __init__(self, **kwargs):
        self.mnist_dataset_defaults.update(**kwargs)
        for key, value in self.mnist_dataset_defaults.items():
            setattr(self, key, value)

        assert self.source_dir is not None
        self.data = input_data.read_data_sets(self.source_dir)
        self.iterator = self.iterator_fn()

    def get_batch(self, batch_size):
        batch_x, batch_y = self.data.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, [batch_size, 28, 28, 1])

        ## move to [-1, 1] for SELU
        batch_x = batch_x * (2) - 1
        return batch_x

    def iterator_fn(self):
        while True:
            batch_x = self.get_batch(self.batch_size)
            # batch_x, batch_y = self.data.train.next_batch(self.batch_size)
            # batch_x = np.reshape(batch_x, [self.batch_size, 28, 28, 1])
            #
            # ## move to [-1, 1] for SELU
            # batch_x = batch_x * (2) - 1
            yield batch_x

    def print_info(self):
        print('---------------------- {} ---------------------- '.format(self.name))
        for key, value in sorted(self.__dict__.items()):
            print('|\t', key, value)
        print('---------------------- {} ---------------------- '.format(self.name))
