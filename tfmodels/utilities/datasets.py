import tensorflow as tf
import numpy as np
import os, glob, cv2
import threading
from openslide import OpenSlide

from tensorflow.examples.tutorials.mnist import input_data

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

""" Using TF's built in MNIST dataset

https://stackoverflow.com/questions/43231958/filling-queue-from-python-iterator
"""
class IteratorDataSet(object):
    mnist_dataset_defaults = {
        'batch_size': 64,
        'capacity': 256,
        'mode': 'TRAIN',
        'name': 'IteratorDataSet',
        'n_classes': 10,
        'source_dir': None,
    }

    def __init__(self, **kwargs):
        self.mnist_dataset_defaults.update(**kwargs)
        for key, value in self.mnist_dataset_defaults.items():
            setattr(self, key, value)

        assert self.source_dir is not None
        self.data = input_data.read_data_sets(self.source_dir)
        self.iterator = self.iterator_fn()

        # self.queue = tf.FIFOQueue(
        #     capacity=self.capacity,
        #     dtypes=[tf.float32] )
        #
        # ## A bit different from below. The equeue_op pulls data from the iterator
        # ## It doesn't need shape??
        # self.batch_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # self.enqueue_op = self.queue.enqueue(self.batch_x)
        # # self.enqueue_op = tf.Print(self.enqueue_op, ['enqueue'])
        # self.image_op = self.queue.dequeue()
        # # self.image_op = tf.Print(self.image_op, ['dequeue'])


    def iterator_fn(self):
        while True:
            batch_x, batch_y = self.data.train.next_batch(self.batch_size)
            batch_x = np.reshape(batch_x, [self.batch_size, 28, 28, 1])
            yield batch_x


    # def enqueue_thread(self):
    #     with self.coord.stop_on_exception():
    #         while not self.coord.should_stop():
    #             self.sess.run(self.enqueue_op,
    #                 feed_dict={self.batch_x: list(next(self.iterator))})
    #
    #
    # def start_enqueue(self):
    #     print 'Setting up threads'
    #     for i in range(self.threads):
    #         threading.Thread(target=self.enqueue_thread).start()



    def print_info(self):
        print '---------------------- {} ---------------------- '.format(self.name)
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '---------------------- {} ---------------------- '.format(self.name)



'''
Implements a threaded queue for reading images from disk given filenames
Since this is for segmentation, we have to also read masks in the same order

Assume the images and masks are named similarly and are in different folders
'''

class DataSet(object):
    defaults = {
        'capacity': 5000,
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
        print '------------------------ DataSet ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ DataSet ---------------------- '


    def _preprocessing(self, image, mask):
        raise Exception(NotImplementedError)


    def get_batch(self):
        raise Exception(NotImplementedError)


'''
It does not work with asynchronous loading.
The two file readers in different streams WILL become desynced,
resulting in mismatches.

We could have a buffer queue for reading-preprocessing,
and an outward facing queue for feeding data to the network.
Alternatively, the image-mask file names should be married together from the
very beginning.

I don't know how to do simple string concat + split in TF ops, so....
'''
class ImageMaskDataSet(DataSet):
    defaults = {
        'batch_size': 16,
        'crop_size': 256,
        'channels': 3,
        'image_dir': None,
        'image_ext': 'jpg',
        'mask_dir': None,
        'mask_ext': 'png',
        'ratio': 1.0,
        'capacity': 5000,
        'seed': 5555,
        'threads': 4,
        'input_size': 1200,
        'min_holding': 1250,
        'dstype': 'ImageMask',
        'augmentation': None }
    def __init__(self, **kwargs):
        self.defaults.update(kwargs)
        # print self.defaults
        super(ImageMaskDataSet, self).__init__(**self.defaults)
        assert self.image_dir is not None
        assert self.mask_dir is not None

        ## ----------------- Load Image Lists ------------------- ##
        image_list = sorted(glob.glob(os.path.join(self.image_dir, '*.'+self.image_ext) ))
        mask_list = sorted(glob.glob(os.path.join(self.mask_dir, '*.'+self.mask_ext) ))

        # for imgname, maskname in zip(image_list, mask_list):
        #     print imgname, maskname

        self.image_names = tf.convert_to_tensor(image_list)
        self.mask_names  = tf.convert_to_tensor(mask_list)

        ## ----------------- Queue ops to feed ----------------- ##
        self.feature_queue = tf.train.string_input_producer(self.image_names,
            shuffle=False,
            seed=self.seed)
        self.mask_queue    = tf.train.string_input_producer(self.mask_names,
            shuffle=False,
            seed=self.seed)

        ## ----------------- TensorFlow ops ------------------- ##
        self.image_reader = tf.WholeFileReader()
        # self.mask_reader = tf.WholeFileReader()

        image_key, image_file = self.image_reader.read(self.feature_queue)
        mask_key, mask_file = self.image_reader.read(self.mask_queue)

        image_file = tf.Print(image_file, [image_key, mask_key])

        image = tf.image.decode_image(image_file)
        mask = tf.image.decode_image(mask_file)

        image, mask = self._preprocessing(image, mask)

        self.image_op, self.mask_op = tf.train.batch([image, mask],
            batch_size = self.batch_size,
            capacity   = self.capacity,
            num_threads = self.threads,
            shapes = [
                [self.input_size, self.input_size, self.channels],
                [self.input_size, self.input_size, 1]],
            name = 'Dataset')
        print 'self.image_op', self.image_op.get_shape()
        print 'self.mask_op', self.mask_op.get_shape()


        self.mask_op = tf.cast(self.mask_op, tf.uint8)
        self.image_shape = self.image_op.get_shape()
        self.mask_shape = self.mask_op.get_shape()


    def _preprocessing(self, image, mask):
        # image = tf.divide(image, 255)
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        image_mask = tf.concat([image, mask], -1)



        ## Cropping
        if self.augmentation == 'random':
            image_mask = tf.random_crop(image_mask,
                [-1, self.crop_size, self.crop_size, 4])
            image_mask = tf.image.random_flip_left_right(image_mask)
            image_mask = tf.image.random_flip_up_down(image_mask)
            image, mask = tf.split(image_mask, [3,1], axis=-1)

            image = tf.image.random_brightness(image, max_delta=0.01)
            # image = tf.image.random_contrast(image, lower=0.7, upper=0.9)
            image = tf.image.random_hue(image, max_delta=0.01)
            image = tf.image.random_saturation(image, lower=0.85, upper=1.0)
        else:
            image, mask = tf.split(image_mask, [3,1], axis=-1)

        ## Resize ratio
        target_h = tf.cast(self.crop_size*self.ratio, tf.int32)
        target_w = tf.cast(self.crop_size*self.ratio, tf.int32)
        image = tf.image.resize_images(image, [target_h, target_w])
        mask = tf.image.resize_images(mask, [target_h, target_w], method=1) ## nearest neighbor

        ## Recenter to [-0.5, 0.5] for SELU activations
        image = tf.multiply(image, 2/255.0) - 1

        # image = tf.Print(image, ['image', tf.reduce_min(image), tf.reduce_max(image)])

        return image, mask


    def get_batch(self, sess):
        image, mask = sess.run([self.image_op, self.mask_op])
        return image, mask

    def print_info(self):
        print '------------------------ ImageMaskDataSet ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ ImageMaskDataSet ---------------------- '




class ImageComboDataSet(DataSet):
    defaults = {
        'augmentation': 'random',
        'batch_size': 16,
        'crop_size': 256,
        'channels': 3,
        'dstype': 'ImageMask',
        'image_dir': None,
        'image_ext': 'png',
        'ratio': 1.0,
        'seed': 5555
    }

    def __init__(self, **kwargs):
        self.defaults.update(kwargs)
        # print self.defaults
        super(ImageComboDataSet, self).__init__(**self.defaults)
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
        image_mask = tf.cast(image_mask, tf.float32)

        ## Cropping
        if self.augmentation == 'random':
            image_mask = tf.random_crop(image_mask,
                [self.crop_size, self.crop_size, 4])
            image_mask = tf.image.random_flip_left_right(image_mask)
            image_mask = tf.image.random_flip_up_down(image_mask)
            image, mask = tf.split(image_mask, [3,1], axis=-1)

            # image = tf.multiply(image, 2/255.0) - 1
            image = tf.image.random_brightness(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.65, upper=1.0)
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_saturation(image, lower=0.65, upper=1.0)
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


    def print_info(self):
        print '------------------------ ImageComboDataSet ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ ImageComboDataSet ---------------------- '


'''
'''
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


    def print_info(self):
        print '------------------------ ImageDataSet ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ ImageDataSet ----------------------'



## TODO
class SVSDataSet(DataSet):
    defaults = {
        'batch_size': 16,
        'crop_size': 256,
        'svs_path': None,
        'ratio': 1.0,
        'capacity': 5000,
        'threads': 4,
        'min_holding': 1250,
        'dstype': 'ImageMask' }
    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
        super(SVSDataSet, self).__init__(**self.defaults)

        assert self.svs_path is not None and os.path.exists(self.svs_path)

        self.svs = OpenSlide(self.svs_path)

        ## Compute a foreground mask

        ## Somehow generate tiles from the file from area underneath the mask
