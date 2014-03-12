#! /bin/bash

fast_copy() {
    if [ -e "$2" ]; then
        # If destination exists, use rsync
        rsync -aW "$1"/ "$2"/
    else
        # If it's a fresh copy, just use cp (2x as fast)
        cp -arL "$1" "$2"
    fi
}

echo 
echo "Beginning training."
echo "  Script:   `basename $0`"
echo "  Date:     `date`"
echo "  Hostname: `hostname`"
echo "  nvidia-smi:"
echo
nvidia-smi | sed 's/^/    /g'
echo
echo "  Copying files..."
fast_copy imagenet_val_leveldb.pristine   imagenet_val_leveldb
fast_copy imagenet_train_leveldb.pristine imagenet_train_leveldb
echo "    ...done."
echo "  Files here:"
echo
ls -ralt | sed 's/^/    /g'
echo
echo "-------------------------------------------------"

GLOG_logtostderr=1 time ./train_net.bin imagenet_solver.prototxt 2>&1
code="$?"

echo "-------------------------------------------------"
echo "Finished (exit code: $code)."

