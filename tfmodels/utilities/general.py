import numpy as np
import cv2
import os, glob
import datetime


""" Save a 4D stack of images
"""
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


def bayesian_inference(model, x_in, samples, keep_prob=0.5):
    ## check x_in for shape
    assert len(x_in.shape) == 4
    assert x_in.shape[0] == 1

    y_hat = model.inference(x_in=x_in, keep_prob=keep_prob)
    y_hat = np.expand_dims(y_hat, -1)
    for tt in xrange(1, samples):
        y_hat_p = model.inference(x_in=x_in, keep_prob=keep_prob)
        y_hat = np.concatenate([y_hat, np.expand_dims(y_hat_p, -1)], -1)

    y_bar_mean = np.mean(y_hat, axis=-1)
    y_bar_var = np.var(y_hat, axis=-1)
    y_bar = np.argmax(y_bar_mean, axis=-1) ## (1, h, w)

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

        img_mask = np.concatenate([img_, mask_], axis=-1)

        success = cv2.imwrite(outname, img_mask)
        print counter, img_mask.shape, img_mask.dtype, outname
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
Convenience function to initialize an experiment
"""
def init_experiment(exp_name, recreate=False, root_dir='.', subdirs=None):
    if subdirs is None:
        subdirs = ['logs', 'snapshots', 'debug', 'inference']

    exp_root = os.path.join(root_dir, exp_name)
    if not os.path.exists(exp_root, 'dir'):
        os.mkdirs(exp_root)


    elif os.path.exists(exp_root, 'dir') and recreate:
        print 'Deleting {}'.format(exp_root)
        shutil.rmdtree(exp_root)
        print 'Remaking {}'.format(exp_root)
