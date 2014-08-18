#! /usr/bin/env ipythonpl

import cv2
from pylab import *

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

from get_net import net



def safe_predict(net, image):
    im_255 = array(image, 'float32', copy=True)
    im_255 -= im_255.min()
    im_255 *= (255.0 / (im_255.max() + 1e-6))
    #print 'im_25 min', im_255.min(), 'max', im_255.max()
    #min_val = image.min()
    #max_val = image.max()
    #if min_val < 0 or min_val > 1 or max_val < 0 or max_val > 1:
    #    print 'WARNING: safe_predict expected image in [0,1] but got range [%g,%g]' % (min_val, max_val)
    return net.predict([im_255], oversample=False)



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
    #net = get_net()
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
            safe_predict(net, im_small)
            print
            clf()
            plot_probs(net, labels)
            draw()
            #print frame.mean()
        rval, frame = cap.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def classit():
    #net = get_net()
    labels = get_labels()

    #filename = caffe_root + 'examples/images/cat.jpg'
    filename = '/Users/jason/Desktop/flower.png'
    img = caffe.io.load_image(filename)
    print 'min', img.min(), 'max', img.max()
    scores = safe_predict(net, img)

    show_net(net)
    clf()
    plot_probs(net, labels)
    


def main():
    cam_loop()
    classit()
    raw_input('enter to exit')



if __name__ == '__main__':
    main()
