#! /usr/bin/env ipythonpl

import cv2
from pylab import *

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe



net = None
imagenet_mean = load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')

def get_net():
    global net
    if net is None:
        net = caffe.Classifier(caffe_root + 'jason/imagenet_deploy.prototxt',
                               #caffe_root + 'jason/140311_234854_afadfd3_priv_netbase/caffe_imagenet_train_iter_450000',
                               caffe_root + 'jason/caffe_reference_imagenet_model',
        )
        net.set_phase_test()
        net.set_mode_gpu()
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        net.set_mean('data', load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
        #net.set_raw_scale('data', scale)  # the reference model operates on images in [0,255] range instead of [0,1]
        net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    return net


    
def get_labels():
    # load labels
    imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    labels = loadtxt(imagenet_labels_filename, str, delimiter='\t')
    return labels

    

def plot_probs(net, labels):
    # sort top k predictions from softmax output
    probs = net.blobs['prob'].data[0].flatten()     # 0 for a single image, not 4 for center crop
    top_k = probs.argsort()[-1:-6:-1]
    #print labels[top_k]
    
    plot(probs)
    for ii, top_x in enumerate(probs.argsort()[-1:-6:-1]):
        print '%d %.03f %3d %s' % (ii, probs[top_x], top_x, labels[top_x])
        plot(top_x, probs[top_x], 'ro')
        text(top_x, probs[top_x], ' %.03f: %s' % (probs[top_x], labels[top_x]))



def show_net(net, layers=False):
    print '%-41s%-31s%s' % ('', 'acts', 'act diffs')
    print '%-45s%-31s%s' % ('', 'params', 'param diffs')
    for k, v in net.blobs.items():
        if k in net.params:
            params = net.params[k]
            for pp, blob in enumerate(params):
                if pp == 0:
                    print '  ', 'P: %-5s'%k,
                else:
                    print ' ' * 11,
                print '%-32s' % repr(blob.data.shape),
                print '%-30s' % ('(%g, %g)' % (blob.data.min(), blob.data.max())),
                print '(%g, %g)' % (blob.diff.min(), blob.diff.max())
        print '%-5s'%k, '%-34s' % repr(v.data.shape),
        print '%-30s' % ('(%g, %g)' % (v.data.min(), v.data.max())),
        print '(%g, %g)' % (v.diff.min(), v.diff.max())

    if layers:
        print
        print 'Layers:'
        print net.complete_layers



def cam_loop():
    net = get_net()
    labels = get_labels()
    cv2.namedWindow('preview')
    cap = cv2.VideoCapture(0)

    #width = cap.get(3)
    #height = cap.get(4)
    #cap.set(3, width / 2.0)
    #cap.set(4, height / 2.0)
    
    rval, frame = cap.read()
    small_shape = (400,400,3)
    #im_small = zeros((400,400,3))

    while True:
        if frame is not None:
            #cv2.resize(frame[:,280:(280+720),:], 0, im_small)
            im_small = cv2.resize(frame[:,280:(280+720),:], small_shape[:2])
            cv2.imshow('preview', im_small)
            net.predict([im_small], oversample=False)
            clf()
            plot_probs(net, labels)
            draw()
            print frame.mean()
        rval, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def classit():
    net = get_net()
    labels = get_labels()

    #filename = caffe_root + 'examples/images/cat.jpg'
    filename = '/Users/jason/Desktop/flower.png'
    img = caffe.io.load_image(filename)
    print 'min', img.min(), 'max', img.max()
    #scores = net.predict([0.00001*img], oversample=False)
    #scores = net.predict([imagenet_mean.transpose((1,2,0))], oversample=False)
    scores = net.predict([imagenet_mean.transpose((1,2,0))[15:15+227,15:15+227]], oversample=False)

    show_net(net)
    clf()
    plot_probs(net, labels)
    


def main():
    cam_loop()
    classit()
    raw_input('enter to exit')



if __name__ == '__main__':
    main()
