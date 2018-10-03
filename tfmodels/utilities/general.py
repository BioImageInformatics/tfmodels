from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import os, glob, sys, shutil
import datetime, time

from datasets import TFRecordImageMask


""" Save a 4D stack of images """
def save_image_stack(stack, writeto,
    prefix='img', ext='jpg', onehot=False, scale='max',
    stack_axis=0):
    assert os.path.exists(writeto)
    n_imgs = stack.shape[stack_axis]
    stack = stack.astype(np.float32)

    ## convert onehot to mask
    if onehot:
        stack=np.argmax(stack, -1).astype(np.float32)

    if scale=='max':
        scaled = stack * 255/stack.max()
    else:
        scaled = stack * 255/scale

    scaled = cv2.convertScaleAbs(scaled)

    img_list = np.split(scaled, n_imgs, stack_axis)
    for nx, img in enumerate(img_list):
        img = np.squeeze(img)
        img_name = '{}/{}_{:04d}.{}'.format(
            writeto, prefix, nx, ext )
        cv2.imwrite(img_name, img)


def bayesian_inference(model, x_in, samples, keep_prob=0.5, verbose=False):
    ## check x_in shape
    assert len(x_in.shape) == 4
    assert x_in.shape[0] == 1
    if verbose:
        print('Entering ops.bayesian_inference() with {} samples'.format(samples))
        print('Got x_in: {} [{}-{}]'.format(x_in.shape, x_in.min(), x_in.max()))
        print('initializing y_hat')

    y_hat = model.inference(x_in=x_in, keep_prob=keep_prob)
    y_hat = np.expand_dims(y_hat, -1)
    if verbose:
        print('y_hat initialized: {}'.format(y_hat.shape))

    for tt in xrange(1, samples):
        y_hat_p = model.inference(x_in=x_in, keep_prob=keep_prob)
        y_hat = np.concatenate([y_hat, np.expand_dims(y_hat_p, -1)], -1)

    if verbose:
        print('y_hat_i: {}'.format(y_hat.shape))
        print('finding means, variance and argmax')
    y_bar_mean = np.mean(y_hat, axis=-1)
    y_bar_var = np.var(y_hat, axis=-1)
    y_bar = np.argmax(y_bar_mean, axis=-1) ## (1, h, w)

    if verbose:
        print('returning mean: {}, var: {}, y_bar: {} [{}]'.format(
            y_bar_mean.shape, y_bar_var.shape, y_bar.shape, np.unique(y_bar)))

    return y_bar_mean, y_bar_var, y_bar


"""
In case images and masks exist as separate files, this script will fuse them
into a 4-channel (RGBA-like) image.
"""
def write_image_mask_combos(img_src_dir, mask_src_dir, save_dir,
    img_src_ext='jpg',
    mask_src_ext='png',
    save_ext='png'):

    img_list = sorted(glob.glob(os.path.join(
        img_src_dir, '*.'+img_src_ext )))
    mask_list = sorted(glob.glob(os.path.join(
        mask_src_dir, '*.'+mask_src_ext )))

    counter = 0
    for img, mask in zip(img_list, mask_list):
        outname = img.replace(img_src_dir, save_dir).replace(img_src_ext, save_ext)
        img_ = cv2.imread(img, -1)
        mask_ = cv2.imread(mask, -1)
        if mask_.shape[-1] == 3:
            mask_ = mask_[:,:,0]
        mask_ = np.expand_dims(mask_, -1)

        if adjust_label:
            mask_ /= adjust_label

        img_mask = np.concatenate([img_, mask_], axis=-1)

        success = cv2.imwrite(outname, img_mask)
        print(counter, img_mask.shape, img_mask.dtype, outname)
        counter += 1


"""
Convenience function for performing a test
Take an image as an iterable

Numpy arrays are iterable along the first dimension
"""
def test_bayesian_inference(model, test_x_list, output_dir, prefix='', keep_prob=0.5, samples=50):
    for test_idx, test_img in enumerate(test_x_list):
        test_img = np.expand_dims(test_img, 0)
        y_bar_mean, y_bar_var = model.bayesian_inference(test_img,
            samples, keep_prob=keep_prob, ret_all=True)
        y_bar = np.argmax(y_bar_mean, axis=-1)
        y_bar = np.expand_dims(y_bar, 0)

        save_image_stack(y_bar, output_dir, prefix='y_bar_{}_{:04d}'.format(prefix, test_idx),
            scale=3, ext='png', stack_axis=0)
        save_image_stack(y_bar_mean, output_dir, prefix='y_mean_{}_{:04d}'.format(prefix, test_idx),
            scale='max', ext='png', stack_axis=-1)
        save_image_stack(y_bar_var, output_dir, prefix='y_var_{}_{:04d}'.format(prefix, test_idx),
            scale='max', ext='png', stack_axis=-1)



"""
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md
"""

def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value]))

def _cut_subimages(img, subimage_size, oversample_factor):
    h, w = img.shape[:2]

    hT = h / float(subimage_size)
    wT = w / float(subimage_size)

    subimgs = []
    for ih in np.linspace(0, h-subimage_size, int(np.ceil(hT)*oversample_factor), dtype=np.int):
        for iw in np.linspace(0, w-subimage_size, int(np.ceil(wT)*oversample_factor), dtype=np.int):
            subimgs.append(img[ih:ih+subimage_size, iw:iw+subimage_size])

    return subimgs

def _read_img_mask(imgp, maskp, img_process_fn=None, mask_process_fn=None, subimage_size=None,
    oversample_factor=1):
    img = cv2.imread(imgp, -1)
    mask = cv2.imread(maskp, -1)

    if img_process_fn is not None:
        img = img_process_fn(img)

    if mask_process_fn is not None:
        mask = mask_process_fn(mask)

    ih, iw = img.shape[:2]
    mh, mw = mask.shape[:2]
    assert ih == mh
    assert iw == mw

    if subimage_size is not None:
        img = _cut_subimages(img, subimage_size, oversample_factor)
        mask = _cut_subimages(mask, subimage_size, oversample_factor)

    return img, mask, ih, iw


def estimate_dataset_size(img_path, mask_path, n_classes, n_examples):
    img_, mask_, ih, iw = _read_img_mask(img_path, mask_path)

    return (img_.nbytes + mask_.nbytes) * n_examples

"""
TODO
Add option to split several records after shuffling to help distributed training
During training we'll have to switch the active dataset with a placeholder
that feeds initialized datasets

(REF: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md)

1. Estimate the dataset size
2. Shuffle the file handles
3. Split it into sizable chunks
4. Save them

Note: via the subimage_size argument this function is set up to handle
variable sized inputs, and normalize them to subimages with uniform size
"""
def image_mask_2_tfrecord(img_patt, mask_patt, record_path, img_process_fn=lambda x: x[:,:,::-1],
    mask_process_fn=None, name_transl_fn=None, n_classes=None, subimage_size=None,
    oversample_factor=1.2, subset_pct=False):

    writer = tf.python_io.TFRecordWriter(record_path)

    # img_list = sorted(glob.glob(os.path.join(img_path, '*'+img_ext)))
    # mask_list = sorted(glob.glob(os.path.join(mask_path, '*'+mask_ext)))
    img_list = sorted(glob.glob(img_patt))
    mask_list = sorted(glob.glob(mask_patt))

    if name_transl_fn is None:
        ## We're relying on our past selves to have not messed up
        assert len(img_list) == len(mask_list)
        assert len(img_list) > 0
    print('Got {} source images'.format(len(img_list)))
    # estimated_size = estimate_dataset_size(img_list[0], mask_list[0], n_classes)

    ## Shuffle and subset
    tmp_list = zip(img_list, mask_list)
    np.random.shuffle(tmp_list)

    if subset_pct:
        assert subset_pct < 1.
        assert subset_pct > 0.
        subset_n = int(len(tmp_list) * float(subset_pct))
        print('Subsetting image list: {} --> {}'.format(len(tmp_list), subset_n))
        tmp_list = tmp_list[:min(len(tmp_list), subset_n)]
        print('Verify new length = {}'.format(len(tmp_list)))

    img_list, mask_list = zip(*tmp_list)

    count = 0
    for source_idx, (imgp, maskp) in enumerate(zip(img_list, mask_list)):
        imgbase = os.path.basename(imgp)
        ## Overwrite mask_list... this is bad bad bad
        if name_transl_fn is not None:
            maskp = name_transl_fn(imgp)
            mask_exists = os.path.exists(maskp)
            if not mask_exists:
                print('WARRNING!! Image {} no matching mask {}'.format(imgp, maskp))
                continue
        maskbase = os.path.basename(maskp)

        imgbase = os.path.splitext(imgbase)[0]
        maskbase = os.path.splitext(maskbase)[0]

        # print imgbase, maskbase
        ## TODO (nathan) check image-masks combos -- names should match
        img, mask, height, width = _read_img_mask(imgp, maskp,
            img_process_fn=img_process_fn,
            mask_process_fn=mask_process_fn,
            subimage_size=subimage_size,
            oversample_factor=oversample_factor)

        if subimage_size is not None:
            for img_, mask_ in zip(img, mask):
                img_raw = img_.tostring()
                mask_raw = mask_.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(subimage_size),
                    'width': _int64_feature(subimage_size),
                    'img': _bytes_feature(img_raw),
                    'mask': _bytes_feature(mask_raw) }))
                writer.write(example.SerializeToString())
                count += 1
                if count % 100 == 0:
                    print('Writing [{}] image [{:05d}] (source [{:05d}]/[{:05d}])'.format(
                        record_path, count, source_idx, len(img_list)))
        else:
            img_raw = img.tostring()
            mask_raw = mask.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'img': _bytes_feature(img_raw),
                'mask': _bytes_feature(mask_raw) }))
            writer.write(example.SerializeToString())
            count += 1
            if count % 100 == 0:
                print('Writing [{}] image [{:05d}]/[{:05d}]'.format(
                    record_path, count, len(img_list)))

    writer.close()
    print('Finished writing [{}]'.format(record_path))


def image_label_2_tfrecord():
    pass


def check_tfrecord(record_path, iterations=25, crop_size=512, image_ratio=0.5,
    batch_size=32, prefetch=500, n_threads=4, as_onehot=True, n_classes=None,
    img_dtype=tf.uint8, mask_dtype=tf.uint8, img_channels=3, mask_channels=1, preprocess=[]):
    with tf.Session() as sess:
        dataset = TFRecordImageMask(
            training_record = record_path,
            as_onehot = as_onehot,
            n_classes = n_classes,
            crop_size = crop_size,
            ratio = image_ratio,
            batch_size = batch_size,
            prefetch = prefetch,
            n_threads = n_threads,
            img_dtype = img_dtype,
            mask_dtype = mask_dtype,
            img_channels = img_channels,
            mask_channels = mask_channels,
            preprocess = preprocess,
            sess = sess )

        pull_times = []
        print('Checking 25 batches of {} examples'.format(batch_size))
        for k in xrange(iterations):
            tstart = time.time()
            img_, mask_ = sess.run([dataset.image_op, dataset.mask_op])
            pull_times.append(time.time() - tstart)
            ps = '{:03d} IMG type [{}] shape [{}]'.format(k, img_.dtype, img_.shape, )
            ps += ' MASK type [{}] range [{}-{}] shape [{}]'.format(mask_.dtype, mask_.min(), mask_.max(), mask_.shape)
            print(ps)

    print('Average time:', np.mean(pull_times))


""" Same as above; except use a dataset defined externally. """
def check_tfrecord_dataset(dataset, iterations=25):
    pull_times = []
    print('Checking average load time for {} batches'.format(iterations))
    for _ in xrange(iterations):
        tstart = time.time()
        img_, mask_ = sess.run([dataset.image_op, dataset.mask_op])
        print(img_.shape, img_.dtype, img_.min(), img_.max(), mask_.dtype, mask_.min(), mask_.max())
        pull_times.append(time.time() - tstart)

    print('Average time:', np.mean(pull_times))


""" Create an experiment directory for organizing logs and snapshots

    log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(basedir)

"""
def make_experiment(basedir, remove_old=False):
    expdate = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir    = os.path.join(basedir, 'logs/{}'.format(expdate))
    save_dir   = os.path.join(basedir, 'snapshots')
    debug_dir  = os.path.join(basedir, 'debug')
    infer_dir  = os.path.join(basedir, 'inference')
    dirlist = [log_dir, save_dir, debug_dir, infer_dir]

    if os.path.isdir(basedir) and remove_old:
        for dd in dirlist:
            if os.path.isdir(dd):
                print('Found directory {}; Cleaning it...'.format(dd))
                shutil.rmtree(dd)
    elif os.path.isdir(basedir) and not remove_old:
        print('Found directory {}; remove_old was False; returning...'.format(basedir))
        return log_dir, save_dir, debug_dir, infer_dir
    else:
        print('Creating base experiment directory')
        os.makedirs(basedir)

    for dd in dirlist:
        print('Creating {}'.format(dd))
        os.makedirs(dd)

    return log_dir, save_dir, debug_dir, infer_dir
