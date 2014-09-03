#! /usr/bin/env python

import sys
from numpy import *
from pascal import *



def main():
    # Script should live in caffe/data/pascal
    directory,script = os.path.split(sys.argv[0])
    pascal_dir = os.path.join(directory, 'VOCdevkit', 'VOC2012')
    image_dir = os.path.join(pascal_dir, '256_JPEGImages')
    annotation_dir = os.path.join(pascal_dir, 'Annotations')
    imsets_dir = os.path.join(pascal_dir, 'ImageSets', 'Main')
    
    print 'Pascal data directory is', pascal_dir
    image_ids = [filename.replace('.jpg','') for filename in os.listdir(image_dir)]
    print 'read', len(image_ids), 'image ids'

    with open(os.path.join(imsets_dir, 'train.txt'), 'r') as ff:
        train_ids = [line.strip() for line in ff.readlines()]
    with open(os.path.join(imsets_dir, 'val.txt'), 'r') as ff:
        val_ids = [line.strip() for line in ff.readlines()]
    with open(os.path.join(imsets_dir, 'trainval.txt'), 'r') as ff:
        trainval_ids = [line.strip() for line in ff.readlines()]

    random.seed(0)
    random.shuffle(train_ids)
    random.seed(1)
    random.shuffle(val_ids)
    random.seed(2)
    random.shuffle(trainval_ids)

    # Tiny versions
    make_pascal('train',    train_ids[:10],    image_dir, annotation_dir, save_as_floats = False, n_per_file = 1000)
    make_pascal('val',      val_ids[:10],      image_dir, annotation_dir, save_as_floats = False, n_per_file = 1000)

    # Whole versions
    #make_pascal('train',    train_ids,    image_dir, annotation_dir, save_as_floats = False, n_per_file = 100000)
    #make_pascal('val',      val_ids,      image_dir, annotation_dir, save_as_floats = False, n_per_file = 100000)




if __name__ == '__main__':
    main()
