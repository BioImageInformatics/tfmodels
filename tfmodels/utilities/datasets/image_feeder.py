from __future__ import print_function
import tensorflow as tf
from .dataset_base import DataSet
import glob, os

"""
Only images, unlabelled
"""
class ImageFeeder(DataSet):

    def __init__(self, **kwargs):
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
        defaults.update(kwargs)
        super(ImageFeeder, self).__init__(**defaults)
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
        print(image.get_shape())

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
