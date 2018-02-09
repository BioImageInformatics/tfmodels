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
class TFRecordImageMask(object):
    defaults = {'training_record': None,
                'testing_record': None,
                'crop_size': 512,
                'ratio': 1.0,
                'batch_size': 32,
                'prefetch': 1000,
                'n_threads': 4,
                'sess': None,
                'as_onehot': True,
                'n_classes': None,
                'img_dtype': tf.uint8,
                'mask_dtype': tf.uint8,
                'img_channels': 3,
                'preprocess': ['brightness', 'hue', 'saturation', 'contrast'],
                'name': 'TFRecordDataset' }
    def __init__(self, **kwargs):
        self.defaults.update(kwargs)

        for key,val in self.defaults.items():
            setattr(self, key, val)

        assert self.training_record is not None

        self.initialized = False

        self.record_path = tf.placeholder_with_default(self.training_record, shape=())
        self.dataset = (tf.data.TFRecordDataset(self.record_path)
                        .repeat()
                        .shuffle(buffer_size=self.batch_size*2)
                        # .prefetch(buffer_size=self.prefetch)
                        .map(lambda x: self._preprocessing(x, self.crop_size, self.ratio),
                            num_parallel_calls=self.n_threads)
                        .prefetch(buffer_size=self.prefetch)
                        .batch(self.batch_size) )

        self.iterator = self.dataset.make_initializable_iterator()
        self.image_op, self.mask_op = self.iterator.get_next()

        if self.sess is not None:
            self._initalize_training(self.sess)

    def _initalize_training(self, sess):
        fd = {self.record_path: self.training_record}
        sess.run(self.iterator.initializer, feed_dict=fd)
        self.phase = 'TRAIN'
        print 'Dataset TRAINING phase'

    def _initalize_testing(self, sess):
        fd = {self.record_path: self.testing_record}
        sess.run(self.iterator.initializer, feed_dict=fd)
        self.phase = 'TEST'
        print 'Dataset TESTING phase'

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
        img = tf.decode_raw(img, self.img_dtype)
        mask = tf.decode_raw(mask, self.mask_dtype)
        # img = tf.image.decode_image(img)
        # mask = tf.image.decode_image(mask)

        img = tf.cast(img, tf.float32)
        mask = tf.cast(mask, tf.float32)

        return height, width, img, mask


    def _preprocessing(self, example, crop_size, ratio):
        h, w, img, mask = self._decode(example)
        img_shape = tf.stack([h, w, self.img_channels], axis=0)
        mask_shape = tf.stack([h, w], axis=0)

        img = tf.reshape(img, img_shape)
        mask = tf.reshape(mask, mask_shape)

        mask = tf.expand_dims(mask, axis=-1)
        image_mask = tf.concat([img, mask], axis=-1)

        image_mask = tf.random_crop(image_mask,
            [crop_size, crop_size, self.img_channels + 1])
        image_mask = tf.image.random_flip_left_right(image_mask)
        image_mask = tf.image.random_flip_up_down(image_mask)
        img, mask = tf.split(image_mask, [self.img_channels,1], axis=-1)

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
        mask = tf.image.resize_images(mask, [target_h, target_w], method=1) ## nearest neighbor

        ## Recenter to [-0.5, 0.5] for SELU activations
        # img = tf.cast(img, tf.float32)
        img = tf.multiply(img, 2/255.0) - 1
        mask = tf.cast(mask, self.mask_dtype)

        if self.as_onehot:
            mask = tf.one_hot(mask, depth=self.n_classes)
            mask = tf.squeeze(mask)
        # mask = tf.reshape(mask,
        #     [-1, self.x_dims[0], self.x_dims[1], self.n_classes])

        return img, mask
