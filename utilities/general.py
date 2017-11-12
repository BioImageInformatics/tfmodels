import numpy as np
import cv2
import os


''' Save a 4D stack of images
'''
def save_image_stack(stack, writeto, prefix='img', ext='jpg', onehot=False, scale='max'):
    assert os.path.exists(writeto)
    n_imgs = stack.shape[0]

    ## convert onehot to mask
    if onehot:
        stack=np.argmax(stack, -1).astype(np.float32)

    ## detect float or uint8
    # if stack.dtype != 'uint8':
    #     print 'Converting non uint'
    #     stack /= stack.max()
    #     stack=cv2.convertScaleAbs(stack)
    #     print '\t stack', stack.shape, stack.dtype, stack.min(), stack.max()
    # try:
    #     stack *= 255.0
    # except:
    #     print 'what'

    if scale=='max':
        stack *= 255/stack.max()
    else:
        stack *= 255/scale

    stack = cv2.convertScaleAbs(stack)
    print '\t {} stack: '.format(prefix), stack.shape, stack.dtype, stack.min(), stack.max()

    for nx in xrange(n_imgs):
        img = np.squeeze(stack[nx,...])
        img_name = '{}/{}_{}.{}'.format(
            writeto, prefix, nx, ext )
        cv2.imwrite(img_name, img)
