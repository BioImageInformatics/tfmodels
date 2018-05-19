# Get the weights of each class:

# f(class) = frequency(class) / < total image area, where class is present >
# weight(class) = (median of f(class)) / f(class)
from __future__ import print_function
import os
import glob
import cv2
import numpy as np
import sys
import argparse

def main(*args):
    imgdir = args[0][0]
    class_num = int(args[0][1])

    print('Looking for {} classes in dir: {}'.format(class_num, imgdir))

    term = os.path.join(imgdir, '*class.png')
    imgs = glob.glob(term)
    print('Found {} images'.format(len(imgs)))

    freqs = np.zeros(shape=(class_num), dtype=np.float)
    counts = np.zeros(shape=(class_num), dtype=np.float)
    present = np.zeros(shape=(class_num), dtype=np.float)
    totalpx = 0.0

    np.random.shuffle(imgs)

    print('Computing f(c) and f_present(c)')
    for index, img in enumerate(imgs):
        im = cv2.imread(img, -1)
        if im.shape[-1] == 3:
            im = im[:,:,0]

        h,w = im.shape[:2]
        ux = np.unique(im)

        if index % 2000 == 0:
            print(index, '/', len(imgs), end='')
            print(img, end='')
            print('{}'.format(im.shape))
            for c in range(class_num):
                print('{}: {} / {}'.format(c, counts[c], present[c]))

        for u in ux:
            counts[u] += (im == u).sum()
            present[u] += (h*w)

        # for u in range(class_num):
        #     img_present = (present[:, u]).sum() * h * w
        #     class_total = (counts[:, u]).sum()
        #
        #     div = class_total / float(img_present)
        #     freqs[0, u] = div
        #
        #     if index % 1000 == 0:
        #         print "f({}) = {}".format(u, div)

    # print 'Computing: {} * {} * {}'.format(index, h, w)
    # totalpx += index * h * w
    # print 'Total px: ', totalpx
    for u in range(class_num):
        freqs[u] = counts[u] / present[u]
        print('Total class {}: {}/{} ({:1.3f}%)'.format(u, counts[u], present[u], freqs[u]*100))

    med_freqs = np.median(freqs)
    weights = np.zeros_like(freqs)

    weightfile = os.path.join(imgdir, 'class_weights.txt')
    with open(weightfile, 'w') as f:
        medf = 'Median frequency: {}\n'.format(med_freqs)
        print(medf)
        f.write(medf)

        for u in range(class_num):
            w = med_freqs / freqs[ u]
            weights[u] = w
            towrite = 'class_weighting: {}'.format(w)
            print(towrite)
            f.write(towrite + '\n')

"""
Get class weighting for masks in a folder. Also write out a file for record.

Usage:
python get_class_weights.py [path] [nclasses]
python get_class_weights.py /path/to/png/ 4
"""
if __name__ == '__main__':

    # parser = argparse.ArgumentParser()

    if len(sys.argv) > 3 or len(sys.argv) < 2:
        print("get_class_weights.py takes exactly two arguments")
    else:
        main(sys.argv[1:])
