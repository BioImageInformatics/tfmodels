from __future__ import print_function 
import tensorflow as tf
import numpy as np

"""
TODO:
https://www.tensorflow.org/programmers_guide/datasets#applying_arbitrary_python_logic_with_tfpy_func

TODO:
We can have multiple initialized datasets sitting around and feed the initializer
into the placeholder via feed_dict:
REF: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md

TFRecordDataset(training_record = None,
    testing_record = None,
    crop_size = 512,
    ratio = 1.0,
    batch_size = 32,
    prefetch = 1000,
    shuffle_buffer = 512,
    n_threads = 4,
    sess = None,
    as_onehot = True,
    n_classes = True,
    img_dtype = tf.uint8,
    mask_dtype = tf.uint8,
    img_channels = 3,
    preprocess = ['brightness', 'hue', 'saturation', 'contrast'],
    name = 'TFRecordDataset' )


"""
class TFRecordImageLabel(object):
    def __init__(self, **kwargs):
        img_label_defaults = {'training_record': None,
                    'testing_record': None,
                    'crop_size': 512,
                    'ratio': 1.0,
                    'batch_size': 32,
                    'prefetch': 1000,
                    'shuffle_buffer': 512,
                    'n_threads': 4,
                    'sess': None,
                    'as_onehot': True,
                    'n_classes': None,
                    'img_dtype': tf.uint8,
                    'img_channels': 3,
                    'preprocess': ['brightness', 'hue', 'saturation', 'contrast'],
                    'name': 'TFRecordImageLabel'
        }
        img_label_defaults.update(kwargs)

        for key,val in img_label_defaults.items():
            setattr(self, key, val)

        assert self.training_record is not None
        assert self.n_classes is not None

        self.initialized = False

        self.record_path = tf.placeholder_with_default(self.training_record, shape=())
        self.dataset = (tf.data.TFRecordDataset(self.record_path)
                        .repeat()
                        .shuffle(buffer_size=self.shuffle_buffer)
                        .map(lambda x: self._preprocessing(x, self.crop_size, self.ratio),
                            num_parallel_calls=self.n_threads)
                        .prefetch(buffer_size=self.batch_size)
                        .batch(self.batch_size)
                        )

        self.iterator = self.dataset.make_initializable_iterator()
        self.image_op, self.label_op = self.iterator.get_next()

        if self.sess is not None:
            self._initalize_training(self.sess)


    def _initalize_training(self, sess):
        fd = {self.record_path: self.training_record}
        _ = sess.run([self.iterator.initializer], feed_dict=fd)
        # sess.run(self.iterator.initializer, feed_dict=fd)
        self.phase = 'TRAIN'
        print('Dataset TRAINING phase')


    def _initalize_testing(self, sess):
        fd = {self.record_path: self.testing_record}
        _ = sess.run([self.iterator.initializer], feed_dict=fd)
        # sess.run(self.iterator.initializer, feed_dict=fd)
        self.phase = 'TEST'
        print('Dataset TESTING phase')


    def print_info(self):
        print('-------------------- {} ---------------------- '.format(self.name))
        for key, value in sorted(self.__dict__.items()):
            print('|\t{}: {}'.format(key, value))
        print('-------------------- {} ---------------------- '.format(self.name))


    def _decode(self, example):
        features = {'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'img': tf.FixedLenFeature((), tf.string, default_value=''),
                    'y': tf.FixedLenFeature((), tf.int64, default_value=0), }
        pf = tf.parse_single_example(example, features)

        height = tf.squeeze(pf['height'])
        width = tf.squeeze(pf['width'])
        label = tf.squeeze(pf['y'])

        img = pf['img']
        img = tf.decode_raw(img, self.img_dtype)

        img = tf.cast(img, tf.float32)
        if self.as_onehot:
            label = tf.one_hot(label, depth=self.n_classes)

        return height, width, img, label


    def _preprocessing(self, example, crop_size, ratio):
        h, w, img, label = self._decode(example)
        img_shape = tf.stack([h, w, self.img_channels], axis=0)

        img = tf.reshape(img, img_shape)

        img = tf.random_crop(img,
            [crop_size, crop_size, self.img_channels])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        for px in self.preprocess:
            if px == 'brightness':
                img = tf.image.random_brightness(img, max_delta=0.05)

            elif px == 'contrast':
                img = tf.image.random_contrast(img, lower=0.7, upper=0.9)

            elif px == 'hue':
                img = tf.image.random_hue(img, max_delta=0.05)

            elif px == 'saturation':
                img = tf.image.random_saturation(img, lower=0.7, upper=0.9)

        target_h = tf.cast(crop_size*ratio, tf.int32)
        target_w = tf.cast(crop_size*ratio, tf.int32)
        img = tf.image.resize_images(img, [target_h, target_w])

        ## Recenter to [-0.5, 0.5] for SELU activations
        # img = tf.cast(img, tf.float32)
        img = tf.multiply(img, 2/255.0) - 1

        return img, label
