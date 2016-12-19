import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
import Image
from skimage.io import imread, imsave
import scipy.io
import os
import argparse

import sys
# Make sure that caffe is on the python path:
caffe_root = '../../caffe'
sys.path.insert(0, caffe_root + 'python')

import caffe

cnn_input_size = 224
crop_height = 224
crop_width = 224
image_mean = np.array((104.00698793, 116.66876762, 122.67891434))

img_data = ['img_000001.jpg', 'img_000002.jpg']


if __name__ == "__main__":
    # remove the following two lines if testing with cpu
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # netowrk specification files
    deploy_file = '../net/conv/deploy.prototxt'
    model_file = '../cachedir/surface_normal_models/best_model.caffemodel'

    # Load the network for testing
    net = caffe.Net(deploy_file, model_file, caffe.TEST)

    for i in range(len(img_data)):
        in_ = imread(os.path.join(os.path.dirname(__file__), img_data[i]))
        in_ = in_.transpose((2, 0, 1))
        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out1 = net.blobs['sigmoid-dsn1'].data[0][0, :, :]
        out2 = net.blobs['sigmoid-dsn2'].data[0][0, :, :]
        out3 = net.blobs['sigmoid-dsn3'].data[0][0, :, :]
        out4 = net.blobs['sigmoid-dsn4'].data[0][0, :, :]
        out5 = net.blobs['sigmoid-dsn5'].data[0][0, :, :]
        fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]

        out_dir = os.path.join(args.output_dir, image_names[idx])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_name1 = os.path.join(out_dir, 'hed_gradient_image_out1.tiff')
        out_name2 = os.path.join(out_dir, 'hed_gradient_image_out2.tiff')
        out_name3 = os.path.join(out_dir, 'hed_gradient_image_out3.tiff')
        out_name4 = os.path.join(out_dir, 'hed_gradient_image_out4.tiff')
        out_name5 = os.path.join(out_dir, 'hed_gradient_image_out5.tiff')
        fuse_name = os.path.join(out_dir, 'hed_gradient_image.tiff')

        print(out_dir)
        imsave(out_name1, out1)
        imsave(out_name2, out2)
        imsave(out_name3, out3)
        imsave(out_name4, out4)
        imsave(out_name5, out5)
        imsave(fuse_name, fuse)
