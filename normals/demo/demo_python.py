import numpy as np
import Image
import os
import math

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
        img = Image.open(os.path.join(os.path.dirname(__file__), img_data[i]))

        img_tmp = img.resize((cnn_input_size, cnn_input_size), resample=Image.LANCZOS)
        img_tmp = np.array(img_tmp) - image_mean
        img_tmp = img_tmp.transpose((2, 0, 1))

        snd = (1 / math.sqrt(3)) * np.ones((cnn_input_size + 200, cnn_input_size + 200, 3))
        depd = np.zeros((cnn_input_size + 200, cnn_input_size + 200))

        net.blobs('data0').reshape((crop_height + 200, crop_width + 200, 3, 1))
        net.blobs('data1').reshape((crop_height + 200, crop_width + 200, 3, 1))
        net.blobs('data2').reshape((crop_height + 200, crop_width + 200, 1, 1))
        import pdb
        pdb.set_trace()

        # run net and take argmax for prediction
        net.forward()
