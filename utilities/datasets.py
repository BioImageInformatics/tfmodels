import tensorflow as tf
import numpy as np
import os, glob, cv2
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
        'capacity': 5000,
        'seed': 5555,
        'threads': 4,
        'min_holding': 1250,}

    def __init__(self, **kwargs):
        for key, value in self.defaults.items():
            setattr(self, key, value)
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

        ## Resize ratio
        target_h = tf.cast(self.crop_size*self.ratio, tf.int32)
        target_w = tf.cast(self.crop_size*self.ratio, tf.int32)
        image = tf.image.resize_images(image, [target_h, target_w])
        mask = tf.image.resize_images(mask, [target_h, target_w], method=1) ## nearest neighbor

        ## Move to [-0.5, 0.5] for SELU activations
        image = tf.multiply(image, 2/255.0) - 1

        # image = tf.Print(image, ['image', tf.reduce_min(image), tf.reduce_max(image)])

        return image, mask

    def _random_normalize(self, image):
        # imgR, imgG, imgB = tf.split(image, 3, -1)
        lab_image = self._rgb_to_lab(image)
        print 'lab_image', lab_image.get_shape()
        imgR, imgG, imgB = tf.split(lab_image, 3, -1)

        meanR, stdR = tf.nn.moments(imgR, axes = [0,1])
        meanG, stdG = tf.nn.moments(imgG, axes = [0,1])
        meanB, stdB = tf.nn.moments(imgB, axes = [0,1])

        frgb = tf.random_normal([3,2], mean=0, stddev=0.001)
        # frgb = tf.Print(frgb, [frgb, meanR, stdR, meanG, stdG, meanB, stdB])
        imgR = (imgR - meanR) / stdR * (stdR + frgb[0,0]) + (meanR + frgb[0,1])
        imgG = (imgG - meanG) / stdG * (stdG + frgb[1,0]) + (meanG + frgb[1,1])
        imgB = (imgB - meanB) / stdB * (stdB + frgb[2,0]) + (meanB + frgb[2,1])

        return tf.concat([imgR, imgG, imgB], -1)

    def get_batch(self, sess):
        image, mask = sess.run([self.image_op, self.mask_op])
        return image, mask

    def print_info(self):
        print '------------------------ ImageMaskDataSet ---------------------- '
        for key, value in sorted(self.__dict__.items()):
            print '|\t', key, value
        print '------------------------ ImageMaskDataSet ---------------------- '




'''
Required:
:image_dir: string
:n_classes: int

The same as ImageMaskDataSet except each image is assumed to exist as a (h,w,4)
where a split --> [3,1] on the 2nd axis will give image, mask

augmentation:
    random: random crop, flip LR, flip UD, coloration
    fixed: no crop, no flip, color standardized to global target
    none: nothing
'''
class ImageComboDataSet(DataSet):
    defaults = {
        'batch_size': 16,
        'crop_size': 256,
        'channels': 3,
        'image_dir': None,
        'image_ext': 'png',
        'ratio': 1.0,
        'dstype': 'ImageMask',
        'augmentation': None }
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
            image = tf.image.random_contrast(image, lower=0.75, upper=1.0)
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_saturation(image, lower=0.75, upper=1.0)

        ## Resize ratio
        target_h = tf.cast(self.crop_size*self.ratio, tf.int32)
        target_w = tf.cast(self.crop_size*self.ratio, tf.int32)
        image = tf.image.resize_images(image, [target_h, target_w])
        mask = tf.image.resize_images(mask, [target_h, target_w], method=1) ## nearest neighbor

        ## Move to [-1, 1] for SELU activations
        image = tf.multiply(image, 2/255.0) - 1

        # image = tf.Print(image, ['image', tf.reduce_min(image), tf.reduce_max(image)])

        return image, mask



    def _random_normalize(self, image):
        # imgR, imgG, imgB = tf.split(image, 3, -1)
        lab_image = self._rgb_to_lab(image)
        print 'lab_image', lab_image.get_shape()
        imgR, imgG, imgB = tf.split(lab_image, 3, -1)

        meanR, stdR = tf.nn.moments(imgR, axes = [0,1])
        meanG, stdG = tf.nn.moments(imgG, axes = [0,1])
        meanB, stdB = tf.nn.moments(imgB, axes = [0,1])

        frgb = tf.random_normal([3,2], mean=0, stddev=0.001)
        # frgb = tf.Print(frgb, [frgb, meanR, stdR, meanG, stdG, meanB, stdB])
        imgR = (imgR - meanR) / stdR * (stdR + frgb[0,0]) + (meanR + frgb[0,1])
        imgG = (imgG - meanG) / stdG * (stdG + frgb[1,0]) + (meanG + frgb[1,1])
        imgB = (imgB - meanB) / stdB * (stdB + frgb[2,0]) + (meanB + frgb[2,1])

        return tf.concat([imgR, imgG, imgB], -1)


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
class ImageDataSet(DataSet):
    defaults = {
        'augmentation': None,
        'batch_size': 16,
        'capacity': 5000,
        'channels': 3,
        'crop_size': 256,
        'dstype': 'Image',
        'image_dir': None,
        'image_ext': 'jpg',
        'input_size': 1200,
        'min_holding': 1250,
        'ratio': 1.0,
        'seed': 5555,
        'threads': 1,
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
        image = self._preprocessing(image)

        self.image_op = tf.train.shuffle_batch(image,
            batch_size = self.batch_size,
            capacity   = self.capacity,
            num_threads = self.threads,
            min_after_dequeue = self.min_holding,
            name = 'Dataset')


    def _preprocessing(self, image):
        image = tf.cast(image, tf.float32)

        ## Cropping
        if self.augmentation == 'random':
            image = tf.random_crop(image_mask,
                [self.crop_size, self.crop_size, 4])
            image = tf.image.random_flip_left_right(image_mask)
            image = tf.image.random_flip_up_down(image_mask)

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
        image, mask = sess.run([self.image_op, self.mask_op])
        return image, mask


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
