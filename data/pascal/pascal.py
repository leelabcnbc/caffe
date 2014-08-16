#! /usr/bin/env python

import os
import h5py
import skimage.io
import xml.etree.ElementTree
from numpy import *

import pdb



labels = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
    ]
label_txt2id = dict([(labels[ii], ii) for ii in range(len(labels))])
n_classes = len(labels)
assert n_classes == 20



def load_int_image(filename):
    '''
    Loads image using skimage, converting from grayscale or alpha as needed. Modified from io.py
    '''
    
    '''
    skimage.io.imread produces an array of shape (height, width, channels)
    In [148]: im = skimage.io.imread('/home/jyosinsk/s/caffe/examples/images/lion.jpg')
    In [149]: im.shape
    Out[149]: (660, 800, 3)
    '''    
    img = skimage.io.imread(filename)
        
    if img.ndim == 2:
        img = tile(img[:, :, newaxis], (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img



def DEP_imagenet_makeit(h5_filename, label_filename, image_dir):
    '''Make an H5 dataset'''
    
    img_filenames = []
    img_labels = []
    with open(label_filename, 'r') as labelfile:
        for line in labelfile:
            filename, label = line.split()
            img_filenames.append(filename)
            img_labels.append(int(label))
    n_images = len(img_filenames)
    print n_images, 'images'
    
    with h5py.File(h5_filename, 'w') as h5:
        if save_as_floats:
            h5.create_dataset('data',
                              (n_images, 3, 256, 256),
                              dtype='float32',
                              chunks = True)
            h5.create_dataset('label',
                              (n_images,),
                              dtype='float32')
        else:
            h5.create_dataset('data',
                              (n_images, 3, 256, 256),
                              dtype='uint8',
                              chunks = True)
            h5.create_dataset('label',
                              (n_images,),
                              dtype='int32')
        data = h5['data']
        labels = h5['label']
        #for ii in xrange(3):
        for ii in xrange(n_images):
            img_filename = img_filenames[ii]
            img_label = img_labels[ii]
            # Use this version for float
            #arr = caffe.io.load_image('%s/%s' % (image_dir, img_filename))
            #data[ii] = arr * 255
            # Use this version for int8
            arr = load_int_image('%s/%s' % (image_dir, img_filename))
            # Transpose from (height, width, channels) to (channels, height, width)
            arr = arr.transpose(2,0,1)
            #print img_filename, img_label, arr.shape
            data[ii] = arr
            labels[ii] = img_label
            if ii%100 == 0:
                print 'Processed', ii, 'so far'
        print 'Finished processing', ii+1, 'files'



def make_pascal(h5_filename_prefix, image_ids, image_dir, annotation_dir, save_as_floats = False, n_per_file = 1000):
    '''Make H5 dataset files from Pascal'''

    n_images = len(image_ids)
    assert n_images > 0

    for file_idx in range(n_images/n_per_file + 1):
        image_ids_this_dataset = image_ids[file_idx*n_per_file:(file_idx+1)*n_per_file]
        n_images_this = len(image_ids_this_dataset)
        if n_images_this == 0: continue   # happens if n_images is a multiple of n_per_file

        h5_filename = h5_filename_prefix + '.%03d.h5' % file_idx
        print 'Creating', h5_filename

        with h5py.File(h5_filename, 'w') as h5:
            if save_as_floats:
                h5.create_dataset('data',
                                  (n_images_this, 3, 256, 256),
                                  dtype='float32',
                                  chunks = True)
                h5.create_dataset('label',
                                  (n_images_this, n_classes),
                                  dtype='float32')
            else:
                h5.create_dataset('data',
                                  (n_images_this, 3, 256, 256),
                                  dtype='uint8',
                                  chunks = True)
                h5.create_dataset('label',
                                  (n_images_this, n_classes),
                                  dtype='int32')
            data = h5['data']
            labels = h5['label']
            #for ii in xrange(3):
            for ii in xrange(n_images_this):
                img_filename = image_dir      + '/' + image_ids_this_dataset[ii] + '.jpg'
                xml_filename = annotation_dir + '/' + image_ids_this_dataset[ii] + '.xml'

                # Labels: -1 for not in image, 0 for only difficult classes in image, 1 for at least one non-difficult class in image.
                this_label = -1 * ones(n_classes)

                with open(xml_filename, 'r') as ff:
                    elems = xml.etree.ElementTree.parse(ff)
                    
                for obj in elems.getroot().findall('object'):
                    #print obj.find('name').text, obj.find('difficult').text
                    class_id = label_txt2id[obj.find('name').text]
                    difficult_txt = obj.find('difficult').text
                    assert difficult_txt in ('0', '1')
                    is_difficult = (difficult_txt == '1')
                    this_value = 0 if is_difficult else 1
                    this_label[class_id] = max(this_label[class_id], this_value)

                arr = load_int_image(img_filename)
                # Transpose from (height, width, channels) to (channels, height, width)
                arr = arr.transpose(2,0,1)
                #print img_filename, img_label, arr.shape
                data[ii] = arr
                labels[ii] = this_label
                if ii > 0 and ii % 100 == 0:
                    print '  ...processed', ii, 'so far'
            print 'Finished processing', ii+1, 'files'



def main():
    pass



if __name__ == '__main__':
    main()
