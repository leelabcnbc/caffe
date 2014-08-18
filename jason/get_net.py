#! /usr/bin/env ipythonpl

from pylab import *
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe



def get_net():
    global net
    if not 'net' in locals():
        net = caffe.Classifier(caffe_root + 'jason/imagenet_deploy.prototxt',
                               caffe_root + 'jason/140311_234854_afadfd3_priv_netbase/caffe_imagenet_train_iter_450000',
                               #caffe_root + 'jason/caffe_reference_imagenet_model',
        )
        net.set_phase_test()
        net.set_mode_gpu()
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        net.set_mean('data', load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
        # We do this in safe_predict instead
        #net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    return net

net = get_net()
