import tensorflow as tf
import numpy as np
import os, glob, cv2
import threading
from openslide import OpenSlide

from tensorflow.examples.tutorials.mnist import input_data


## https://stackoverflow.com/questions/29831489/numpy-1-hot-array
def np_onehot(a, depth):
    b = np.zeros((a.shape[0], depth))
    b[np.arange(a.shape[0]), a] = 1

    return b

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)

""" return 4-D tensor with shape = (batchsize, h, w, channels) """
def load_images(paths, batchsize, crop_size):
    ## use numpy
    tensor = []
    imglist = np.random.choice(paths, batchsize)
    for imgp in imglist:
        # print imgp
        tensor.append(cv2.imread(imgp)[:,:,::-1]) ## BGR --> RGB

    ## Apply a crop
    ## Can't just do it down the stack in case there are repeats
    fx = lambda ix: np.random.randint(ix.shape[0]-crop_size)
    fy = lambda ix: np.random.randint(ix.shape[1]-crop_size)
    for k in range(batchsize):
        xx = fx(tensor[k])
        yy = fx(tensor[k])
        tensor[k] = tensor[k][xx:xx+crop_size, yy:yy+crop_size,:]

    ## Also treat images like they're the same size
    tensor = [np.expand_dims(x,0) for x in tensor]
    tensor = np.concatenate(tensor, 0).astype(np.float32)
    ## TODO: Logic for float-valued image
    # tensor /= tensor.max()
    tensor /= 255.0

    # print 'Loaded {} tensor : {}, {}\t{}'.format(tensor.shape,
    #     tensor.min(), tensor.max(), tensor.dtype)
    return tensor


"""
input paired x, y matrices:
random_mnist.*.images ~ [obs, dims]
random_mnist.*.labels ~ [obs, 10] ## one-hot

output x sorted by y, as dictionaries
"""
# def collect_mnist(mnist):
#     mnist_out = {}
#     for y in range(0, 10):
#         train_x = mnist.train.images[mnist.train.labels==y, :]
#         test_x = mnist.train.images[mnist.test.labels==y, :]
#         mnist_out{y} = np.vstack([train_x, test_x])
#
#     return mnist_out

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

    def iterator_fn(self):
        while True:
            batch_x, batch_y = self.data.train.next_batch(self.batch_size)
            batch_x = np.reshape(batch_x, [self.batch_size, 28, 28, 1])

            ## move to [-1, 1] for SELU
            batch_x = batch_x * (2) - 1
            yield batch_x

    def print_info(self):
        print '---------------------- {} ---------------------- '.format(self.name)
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '---------------------- {} ---------------------- '.format(self.name)



## TODO
"""
Produces batches like:
x = [batch_size, samples, [dimensions]], y = [batch_size]

where y is 0 or 1 for a batch containing the positive class

batches containing the positive class may have a variable number of
positive examples
"""
class BaggedMNIST(object):
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

    def __init__(self, **kwargs):
        print 'Initializing Bagged MNIST dataset'
        self.bag_mnist_defaults.update(**kwargs)
        for key, value in self.bag_mnist_defaults.items():
            setattr(self, key, value)

        ## load without onehot for ez
        # self.data = input_data.read_data_sets(self.source_dir)

        print 'Using positive class:', self.positive_class
        self.negative_x, self.positive_x = _collect_mnist(self.data,
            positive_class=self.positive_class)
        self._count_examples()
        self.iterator = self.iterator_fn()


    def _count_examples(self):
        self.negative_count = self.negative_x.shape[0]
        self.positive_count = self.positive_x.shape[0]

        print 'Got negative examples', self.negative_count
        print 'Got positive examples', self.positive_count


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
        print '---------------------- {} ---------------------- '.format(self.name)
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '---------------------- {} ---------------------- '.format(self.name)


"""
Implements a threaded queue for reading images from disk given filenames
Since this is for segmentation, we have to also read masks in the same order

Assume the images and masks are named similarly and are in different folders
"""
class DataSet(object):
    defaults = {
        'capacity': 5000,
        'name': 'DataSet',
        'seed': 5555,
        'threads': 4,
        'min_holding': 1250,}

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.batch_size >= 1
        assert self.image_dir is not None


    def print_info(self):
        print '-------------------- {} ---------------------- '.format(self.name)
        for key, value in sorted(self.__dict__.items()):
            print '|\t{}: {}'.format(key, value)
        print '-------------------- {} ---------------------- '.format(self.name)


    def _preprocessing(self, image, mask):
        raise Exception(NotImplementedError)


    def get_batch(self):
        raise Exception(NotImplementedError)


class TFRecordImageMask(object):
    defaults = {'record_path': None,
                'crop_size': 512,
                'ratio': 1.0,
                'batch_size': 32,
                'prefetch': 1000,
                'n_threads': 4,
                'sess': None,
                'name': 'TFRecordDataset' }
    def __init__(self, **kwargs):
        self.defaults.update(kwargs)

        for key,val in self.defaults.items():
            setattr(self, key, val)

        assert self.record_path is not None

        self.initialized = False

        self.dataset = (tf.data.TFRecordDataset(self.record_path)
                        .repeat()
                        .shuffle(buffer_size=self.batch_size*3)
                        # .prefetch(buffer_size=self.prefetch)
                        .map(lambda x: self._preprocessing(x, self.crop_size, self.ratio),
                            num_parallel_calls=self.n_threads)
                        .prefetch(buffer_size=self.prefetch)
                        .batch(self.batch_size) )

        self.iterator = self.dataset.make_initializable_iterator()
        self.image_op, self.mask_op = self.iterator.get_next()

        if self.sess is not None:
            self._initalize_iterator(self.sess)

    def _initalize_iterator(self, sess):
        sess.run(self.iterator.initializer)
        self.initialized = True

    def print_info(self):
        print '-------------------- {} ---------------------- '.format(self.name)
        for key, value in sorted(self.__dict__.items()):
            print '|\t{}: {}'.format(key, value)
        print '-------------------- {} ---------------------- '.format(self.name)

    def _decode(self, example):
        features = {'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'img': tf.FixedLenFeature((), tf.string, default_value=''),
                    'mask': tf.FixedLenFeature((), tf.string, default_value=''), }
        pf = tf.parse_single_example(example, features)

        height = tf.squeeze(pf['height'])
        width = tf.squeeze(pf['width'])

        img = pf['img']
        mask = pf['mask']
        img = tf.decode_raw(img, tf.uint8)
        mask = tf.decode_raw(mask, tf.uint8)
        # img = tf.image.decode_image(img)
        # mask = tf.image.decode_image(mask)

        img = tf.cast(img, tf.float32)
        mask = tf.cast(mask, tf.float32)

        return height, width, img, mask


    def _preprocessing(self, example, crop_size, ratio):
        h, w, img, mask = self._decode(example)
        img_shape = tf.stack([h, w, -1], axis=0)
        mask_shape = tf.stack([h, w], axis=0)

        img = tf.reshape(img, img_shape)
        mask = tf.reshape(mask, mask_shape)

        mask = tf.expand_dims(mask, axis=-1)
        image_mask = tf.concat([img, mask], axis=-1)

        image_mask = tf.random_crop(image_mask,
            [crop_size, crop_size, 4])
        image_mask = tf.image.random_flip_left_right(image_mask)
        image_mask = tf.image.random_flip_up_down(image_mask)
        img, mask = tf.split(image_mask, [3,1], axis=-1)

        img = tf.image.random_brightness(img, max_delta=0.1)
        # img = tf.image.random_contrast(img, lower=0.7, upper=0.9)
        img = tf.image.random_hue(img, max_delta=0.1)
        # img = tf.image.random_saturation(img, lower=0.7, upper=0.9)

        target_h = tf.cast(crop_size*ratio, tf.int32)
        target_w = tf.cast(crop_size*ratio, tf.int32)
        img = tf.image.resize_images(img, [target_h, target_w])
        mask = tf.image.resize_images(mask, [target_h, target_w], method=1) ## nearest neighbor

        ## Recenter to [-0.5, 0.5] for SELU activations
        # img = tf.cast(img, tf.float32)
        img = tf.multiply(img, 2/255.0) - 1
        mask = tf.cast(mask, tf.uint8)

        return img, mask


"""
Pre-concatenated image-mask: [h,w,4]
"""
class ImageComboDataSet(DataSet):
    image_combo_defaults = {
        'augmentation': 'random',
        'batch_size': 16,
        'crop_size': 256,
        'channels': 3,
        'dstype': 'ImageMask',
        'image_dir': None,
        'image_ext': 'png',
        'name': 'ImageCombo',
        'ratio': 1.0,
        'seed': 5555
    }

    def __init__(self, **kwargs):
        self.image_combo_defaults.update(kwargs)
        # print self.defaults
        super(ImageComboDataSet, self).__init__(**self.image_combo_defaults)
        assert self.image_dir is not None

        ## ----------------- Load Image Lists ------------------- ##
        image_list = sorted(glob.glob(os.path.join(self.image_dir, '*.'+self.image_ext) ))
        self.image_names = tf.convert_to_tensor(image_list)

        ## ----------------- Queue ops to feed ----------------- ##
        self.feature_queue = tf.train.string_input_producer(self.image_names,
            capacity=self.capacity,
            shuffle=False,
            seed=self.seed)

        self.image_reader = tf.WholeFileReader()
        image_key, image_file = self.image_reader.read(self.feature_queue)
        image = tf.image.decode_image(image_file, channels=4)
        print image.get_shape()

        #with tf.device('/cpu:0'):
        image, mask = self._preprocessing(image)

        self.image_op, self.mask_op = tf.train.shuffle_batch([image, mask],
            batch_size = self.batch_size,
            capacity   = self.capacity,
            num_threads = self.threads,
            min_after_dequeue = self.min_holding,
            name = 'Dataset')

        self.mask_op = tf.cast(self.mask_op, tf.uint8)

    def _preprocessing(self, image_mask):
        with tf.name_scope('preprocessing'):
            image_mask = tf.cast(image_mask, tf.float32)

            ## Cropping
            if self.augmentation == 'random':
                image_mask = tf.random_crop(image_mask,
                    [self.crop_size, self.crop_size, 4])
                image_mask = tf.image.random_flip_left_right(image_mask)
                image_mask = tf.image.random_flip_up_down(image_mask)
                image, mask = tf.split(image_mask, [3,1], axis=-1)

                # image = tf.multiply(image, 2/255.0)-1
                image = tf.image.random_brightness(image, max_delta=0.1)
                # image = tf.image.random_contrast(image, lower=0.7, upper=1.0)
                image = tf.image.random_hue(image, max_delta=0.1)
                # image = tf.image.random_saturation(image, lower=0.7, upper=1.0)
            else:
                image, mask = tf.split(image_mask, [3,1], axis=-1)

            ## Resize ratio
            target_h = tf.cast(self.crop_size*self.ratio, tf.int32)
            target_w = tf.cast(self.crop_size*self.ratio, tf.int32)
            image = tf.image.resize_images(image, [target_h, target_w])
            mask = tf.image.resize_images(mask, [target_h, target_w], method=1) ## nearest neighbor

            ## Recenter to [-1, 1] for SELU activations
            image = tf.multiply(image, 2/255.0) - 1

        # image = tf.Print(image, ['image', tf.reduce_min(image), tf.reduce_max(image)])
        return image, mask


    def get_batch(self, sess):
        image, mask = sess.run([self.image_op, self.mask_op])
        return image, mask


"""
Only images, unlabelled
"""
class ImageFeeder(DataSet):
    defaults = {
        'augmentation': None,
        'batch_size': 16,
        'capacity': 5000,
        'channels': 3,
        'crop_size': 256,
        'dstype': 'Image',
        'image_dir': None,
        'image_ext': 'jpg',
        'min_holding': 1250,
        'name': 'ImageFeeder',
        'ratio': 1.0,
        'seed': 5555,
        'threads': 1,
    }

    def __init__(self, **kwargs):
        self.defaults.update(kwargs)
        # print self.defaults
        super(ImageFeeder, self).__init__(**self.defaults)
        assert self.image_dir is not None

        ## ----------------- Load Image Lists ------------------- ##
        image_list = sorted(glob.glob(os.path.join(self.image_dir, '*.'+self.image_ext) ))
        self.image_names = tf.convert_to_tensor(image_list)

        ## ----------------- Queue ops to feed ----------------- ##
        self.feature_queue = tf.train.string_input_producer(self.image_names,
            shuffle=False,
            seed=self.seed)

        self.image_reader = tf.WholeFileReader()
        image_key, image_file = self.image_reader.read(self.feature_queue)
        image = tf.image.decode_image(image_file, channels=self.channels)
        print image.get_shape()

        #with tf.device('/cpu:0'):
        image = self._preprocessing(image)

        self.image_op = tf.train.shuffle_batch([image],
            batch_size = self.batch_size,
            capacity   = self.capacity,
            num_threads = self.threads,
            min_after_dequeue = self.min_holding,
            name = 'Dataset')


    def _preprocessing(self, image):
        image = tf.cast(image, tf.float32)

        ## ????
        image = tf.random_crop(image,
            [self.crop_size, self.crop_size, self.channels])

        ## Cropping
        if self.augmentation == 'random':
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

            image = tf.image.random_brightness(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.0)
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_saturation(image, lower=0.75, upper=1.0)

        ## Resize ratio
        target_h = tf.cast(self.crop_size*self.ratio, tf.int32)
        target_w = tf.cast(self.crop_size*self.ratio, tf.int32)
        image = tf.image.resize_images(image, [target_h, target_w])

        ## Move to [-1, 1] for SELU activations
        image = tf.multiply(image, 2/255.0) - 1

        return image


    def get_batch(self, sess):
        image = sess.run([self.image_op])[0]
        return image



## TODO
class SVSDataSet(DataSet):
    svs_defaults = {
        'batch_size': 16,
        'crop_size': 256,
        'svs_path': None,
        'ratio': 1.0,
        'capacity': 5000,
        'threads': 4,
        'min_holding': 1250,
        'dstype': 'ImageMask' }
    def __init__(self, **kwargs):
        self.svs_defaults.update(**kwargs)
        super(SVSDataSet, self).__init__(**self.svs_defaults)

        assert self.svs_path is not None and os.path.exists(self.svs_path)

        self.svs = OpenSlide(self.svs_path)

        ## Compute a foreground mask

        ## Somehow generate tiles from the file from area underneath the mask
