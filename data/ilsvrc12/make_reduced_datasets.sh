#! /bin/bash

# Creates all reduced datasets

#echo "Remove this line if you're sure you want to run (overwrites some data files)" && exit 1

make_half ()
{
    seed="$1"
    tvt="$2"
    ./make_reduced_dataset.py --seed ${seed} data/whole_${tvt}/files.txt
    mkdir -p data/half${seed}A_${tvt}
    mkdir -p data/half${seed}B_${tvt}
    mv reduced_A.txt        data/half${seed}A_${tvt}/files.txt
    mv reduced_A_idxmap.txt data/half${seed}A_${tvt}/idxmap.txt
    mv reduced_B.txt        data/half${seed}B_${tvt}/files.txt
    mv reduced_B_idxmap.txt data/half${seed}B_${tvt}/idxmap.txt
} 

make_half_tvt ()
{
    make_half $1 train
    make_half $1 valid
    make_half $1 test
}

make_half_tvt 0
make_half_tvt 1
make_half_tvt 2
make_half_tvt 3
make_half_tvt 4
make_half_tvt 5
make_half_tvt 6
make_half_tvt 7
make_half_tvt 8
make_half_tvt 9
