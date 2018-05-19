from __future__ import print_function
import numpy as np
"""
Accepts numpy arrays

Stack --> square
[N, h, w, c]

TODO add support for non-square images
"""
def vis_square(images, ht=None, pad=2):
    assert ht

    if len(images.shape) == 3:
        images = np.expand_dims(images, -1)

    if pad:
        ht += pad*2
        padtuple = ((0,0), (pad,pad), (pad,pad), (0,0))
        images = np.pad(images, padtuple, mode='constant',
            constant_values=0.0)

    ## TODO add support non-perfect squares
    n_side_sq = images.shape[0]
    img_size = images.shape[1]
    n_side = int(np.sqrt(n_side_sq))
    n_use = np.square(n_side)

    img_out = np.zeros([(n_side)*ht, (n_side)*ht], dtype=np.float32)

    ix = 0
    if img_size != ht:
        for k in xrange(n_side):
            for kk in xrange(n_side):
                sq = [(k)*ht, (k+1)*ht, (kk)*ht, (kk+1)*ht]
                img = np.squeeze(images[ix,...])
                img = cv2.resize(img, dsize=(ht, ht))
                img_out[sq[0]:sq[1], sq[2]:sq[3]] = img
                ix+=1
    else:
        for k in xrange(n_side):
            for kk in xrange(n_side):
                sq = [(k)*ht, (k+1)*ht, (kk)*ht, (kk+1)*ht]
                img_out[sq[0]:sq[1], sq[2]:sq[3]] = np.squeeze(images[ix,...])
                ix+=1

    return img_out*255


"""
Dream a manifold varying z
"""
def dream_manifold(model, minmax=[-2.0, 2.0], zx=None, n_samples=121, condition=None,
    z_manifold_in=None):

    if z_manifold_in is not None:
        try:
            return vis_square(model.inference(z_values=z_manifold_in),
                ht=model.x_dims[0])
        except:
            print('z_manifold_in appears to have disagreed with the model')

    if zx is not None:
    ## Vary a particular variable, keep the rest at 0.0
        z_manifold = np.zeros([n_samples, model.z_dim])
        z_manifold[:, zx] = np.linspace(minmax[0], minmax[1], n_samples)
    else:
    ## Vary the
        z_manifold = np.random.randn(n_samples, model.z_dim)

    # if condition is not None:
    #     test_y = np.zeros([n_samples, y_dim]).astype(np.float32)
    #     test_y[:, condition] = 1.0
    # else:
    #     test_y = np.zeros([n_samples, y_dim]).astype(np.float32)
    #     test_y[:, np.random.randint(y_dim)] = 1.0
    # test_feed_dict = {z_in: z_manifold, y_in: test_y}
    # x_dream = model.inference(z_values=z_manifold)

    return vis_square(model.inference(z_values=z_manifold), ht=model.x_dims[0])
