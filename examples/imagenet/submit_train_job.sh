#! /bin/bash

# Exit on errors
set -e

if [ "$GIT_RESULTS_MANAGER_DIR" = "" ]; then
    echo "You must run this from within GitResultsManager, something like this:"
    echo "   resman -n `basename $0`"
    exit 1
fi

dir="$GIT_RESULTS_MANAGER_DIR"

if [ "$1" = "" -o "$2" = "" -o "$3" = "" ]; then
    echo "You must provide three arguments: traindb valdb meanfile, something like this:"
    echo "   resman -n `basename $0` imagenet_train_leveldb imagenet_val_leveldb ../../data/ilsvrc12/FETCHED_imagenet_mean.binaryproto"
    exit 1
fi

train_leveldb="$1"
val_leveldb="$2"
mean="$3"

echo "Directory: $dir"

files="../../build/tools/train_net.bin _train_start.sh _train_resume.sh imagenet_solver.prototxt imagenet_train.prototxt imagenet_val.prototxt"

echo
echo "Copying required files:"
for file in $files; do
    echo -n "  "
    cp -v $file $dir/
done
echo -n "  "
cp -v "$mean" $dir/imagenet_mean.binaryproto

#echo
#echo "Hardlinking train and val leveldbs:"
#echo -n "  "
#ln -sv "`pwd`/$train_leveldb" $dir/imagenet_train_leveldb
#echo -n "  "
#ln -sv "`pwd`/$val_leveldb" $dir/imagenet_val_leveldb

echo "Linking train and val leveldbs:"
echo -n "  "
ln -sv "`pwd`/$train_leveldb" $dir/imagenet_train_leveldb.pristine
echo -n "  "
ln -sv "`pwd`/$val_leveldb" $dir/imagenet_val_leveldb.pristine

echo "Submitting job..."

# jobname is formed by shortening something like
# "results/140311_212922_18c4d2f_priv_junk2"
# to this
# "140311_212922_junk2"
jobname="`basename \"$dir\" | gawk -F _ '{print $1 \"_\" $2 \"_\" $5}'`"

qsub -A EvolvingAI -l nodes=1:ppn=4:gpus=1:k20 -l walltime=168:00:00 -N "$jobname" -d "$dir" ./_train_start.sh
