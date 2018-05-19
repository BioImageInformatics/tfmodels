from __future__ import print_function
import tensorflow as tf
import glob

from .dataset_base import DataSet


"""
Pre-concatenated image-mask: [h,w,4]

this is old --- it might be useful again one day

"""
class ImageComboDataSet(DataSet):

    def __init__(self, **kwargs):
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
        image_combo_defaults.update(kwargs)
        super(ImageComboDataSet, self).__init__(**image_combo_defaults)
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
        image_mask = tf.image.decode_image(image_file, channels=4)
        print('image_mask shape:', image_mask.get_shape())

        #with tf.device('/cpu:0'):
        image, mask = self._preprocessing(image_mask)

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
