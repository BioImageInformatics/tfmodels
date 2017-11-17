import numpy as np
import cv2
import os


''' Save a 4D stack of images
'''
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
