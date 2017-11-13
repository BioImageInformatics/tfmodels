import tensorflow as tf
import numpy as np
import os, glob, cv2

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


class MNISTDataSet(object):
    def __init__(self,
                 source_dir,
                 batch_size,
                 n_classes = 10,
                 mode='TRAIN'):

        self.mnist = input_data.read_data_sets(source_dir)
        self.has_masks = False
        self.batch_size = batch_size
        self.mode = mode
        self.use_feed = True

        ## Don't need that fancy stuff down there

        # if self.mode=='TRAIN':
        # self.image_op = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='MNIST_x')
        # self.image_op = tf.cast(self.image_op_vec, tf.float32)
            # self.image_op, self.labels_op = self.mnist.train.next_batch(self.batch_size)
        # elif self.mode=='TEST':
        # self.image_op, self.labels_op = self.mnist.test.next_batch(self.batch_size)

        # self.image_op = self._reshape_batch(self.image_op_vec)

    ## Dummy method -
    def set_tf_sess(self, sess):
        return


    def _reshape_batch(self, vect_x):
        dims = [self.batch_size, 28, 28, 1]
        batch = np.reshape(vect_x, dims)
        return batch


    ## Get a mean image to subtract
    def _compute_mean(self):
        pass


'''
Implements a threaded queue for reading images from disk given filenames
Since this is for segmentation, we have to also read masks in the same order

Assume the images and masks are named similarly and are in different folders
'''

class DataSet(object):
    defaults = {
        'batch_size': 64,
        'crop_size': 256,
        'ratio': 1.0,
        'capacity': 5000,
        'seed': 5555,
        'threads': 4,
        'min_holding': 1250,}

    def __init__(self, **kwargs):
        for key, value in self.defaults.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

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
Required:
:image_dir: string
:mask_dir: string
:n_classes: int
'''
class ImageMaskDataSet(DataSet):
    defaults = {
        'batch_size': 16,
        'crop_size': 256,
        'image_dir': None,
        'image_ext': 'jpg',
        'mask_dir': None,
        'mask_ext': 'png',
        'ratio': 1.0,
        'capacity': 5000,
        'seed': 5555,
        'threads': 4,
        'min_holding': 1250,
        'dstype': 'ImageMask' }
    def __init__(self, **kwargs):
        self.defaults.update(kwargs)
        # print self.defaults
        super(ImageMaskDataSet, self).__init__(**self.defaults)
        assert self.image_dir is not None
        assert self.mask_dir is not None

        ## ----------------- Load Image Lists ------------------- ##
        self.image_names = tf.convert_to_tensor(sorted(glob.glob(
            os.path.join(self.image_dir, '*.'+self.image_ext) )))
        self.mask_names  = tf.convert_to_tensor(sorted(glob.glob(
            os.path.join(self.mask_dir, '*.'+self.mask_ext) )))

        ## ----------------- Queue ops to feed ----------------- ##
        self.feature_queue = tf.train.string_input_producer(
            self.image_names,
            shuffle=True,
            seed=self.seed )
        self.mask_queue    = tf.train.string_input_producer(
            self.mask_names,
            shuffle=True,
            seed=self.seed )

        ## ----------------- TensorFlow ops ------------------- ##
        self.image_reader = tf.WholeFileReader()
        self.mask_reader = tf.WholeFileReader()

        image_key, image_file = self.image_reader.read(self.feature_queue)
        image_op = tf.image.decode_image(image_file, channels=3)

        mask_key, mask_file = self.image_reader.read(self.mask_queue)
        # if self.mask_ext=='png':
        #     mask_op = tf.image.decode_png(mask_file)
        # else:
        mask_op = tf.image.decode_image(mask_file)

        image_op, mask_op = self._preprocessing(image_op, mask_op)
        image_op, mask_op = tf.train.shuffle_batch([image_op, mask_op],
            batch_size = self.batch_size,
            capacity   = self.capacity,
            min_after_dequeue = self.min_holding,
            num_threads = self.threads,
            name = 'Dataset')

        self.image_op = image_op
        self.mask_op = tf.cast(mask_op, tf.uint8)

        self.image_shape = self.image_op.get_shape()
        self.mask_shape = self.mask_op.get_shape()


    def _preprocessing(self, image, mask):
        image = tf.divide(image, 255)
        mask = tf.cast(mask, tf.float32)
        # mask = tf.divide(mask, 255)
        image_mask = tf.concat([image, mask], -1)
        # input_dim = tf.shape()
        # last_dim = tf.shape(image_mask)[-1]

        ## Cropping
        image_mask = tf.random_crop(image_mask,
            [self.crop_size, self.crop_size, 4])
        image, mask = tf.split(image_mask, [3,1], axis=-1)

        ## Resize ratio
        target_h = tf.cast(self.crop_size*self.ratio, tf.int32)
        target_w = tf.cast(self.crop_size*self.ratio, tf.int32)
        image = tf.image.resize_images(image, [target_h, target_w])
        mask = tf.image.resize_images(mask, [target_h, target_w],
            method=1) ## nearest neighbor

        return image, mask


    def get_batch(self, sess):
        image, mask = sess.run([self.image_op, self.mask_op])
        return image, mask


    def print_info(self):
        print '------------------------ ImageMaskDataSet ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ ImageMaskDataSet ---------------------- '


# """ Same as ImageMaskDataSet except images only """
# class ImageDataSet(object):
#     def __init__(self,
#                  image_dir,
#                  n_classes  = 2,
#                  batch_size = 96,
#                  crop_size  = 256,
#                  ratio      = 1.0, ## for future
#                  capacity   = 2000,
#                  image_ext  = 'jpg',
#                  seed       = 5555,
#                  threads    = 4,
#                  min_holding= 250):
#
#         self.image_names = tf.convert_to_tensor(sorted(glob.glob(
#         os.path.join(image_dir, '*.'+image_ext) )))
#         print '{} image files from {}'.format(self.image_names.shape, image_dir)
#
#         self.batch_size = batch_size
#         self.crop_size  = crop_size
#         self.ratio = ratio
#         self.capacity  = capacity
#         self.n_classes = n_classes
#         self.threads = threads
#         self.image_ext = image_ext
#         self.min_holding = min_holding
#         self.has_masks = False
#         self.use_feed = False
#
#         self.preprocess_fn = self._preprocessing
#
#         self.feature_queue = tf.train.string_input_producer(
#             self.image_names,
#             shuffle=True,
#             seed=seed )
#
#         self.image_reader = tf.WholeFileReader()
#         self._setup_image_mask_ops()
#
#
#
#     def set_tf_sess(self, sess):
#         self.sess = sess
#
#
#     def _setup_image_mask_ops(self):
#         print 'Setting up image and mask retrieval ops'
#         with tf.name_scope('ImageDataSet'):
#             image_key, image_file = self.image_reader.read(self.feature_queue)
#             image_op = tf.image.decode_image(image_file)
#
#             image_op = self.preprocess_fn(image_op)
#             image_op = tf.train.shuffle_batch([image_op],
#                 batch_size = self.batch_size,
#                 capacity   = self.capacity,
#                 min_after_dequeue = self.min_holding,
#                 num_threads = self.threads,
#                 name = 'Dataset')
#
#             self.image_op = image_op
#
#
#     def _preprocessing(self, image):
#         ## TODO: setup preprocessing via input_fn
#         image = tf.divide(image, 255)
#
#         ## Perform a random crop
#         image = tf.random_crop(image,
#             [self.crop_size, self.crop_size, 3])
#
#         return image
