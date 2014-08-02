#!/usr/bin/env sh
# Create the imagenet leveldb inputs
# N.B. set the path to the imagenet train + val data dirs

TOOLS=../../build/tools
DATA=../../data/ilsvrc12

echo "Creating leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $HOME/imagenet2012/train/ \
    $DATA/train.txt \
    ilsvrc12_train_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $HOME/imagenet2012/val/ \
    $DATA/val.txt \
    imagenet_val_leveldb 1
    #/home/jyosinsk/imagenet2012/imagenet_val_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $HOME/imagenet2012/test/ \
    $DATA/test.txt \
    imagenet_test_leveldb 1
    #/home/jyosinsk/imagenet2012/imagenet_test_leveldb 1

echo "Done."
