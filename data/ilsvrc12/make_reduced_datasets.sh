#! /bin/bash

# Creates all reduced datasets

echo "Remove this line if you're sure you want to run (overwrites some data files)" && exit 1

set -e

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

make_half_from_files ()
{
    fileA="$1"
    fileB="$2"
    name="$3"
    tvt="$4"
    ./make_reduced_dataset.py --half-files "$fileA" "$fileB" data/whole_${tvt}/files.txt
    mkdir -p data/half${name}A_${tvt}
    mkdir -p data/half${name}B_${tvt}
    mv reduced_A.txt        data/half${name}A_${tvt}/files.txt
    mv reduced_A_idxmap.txt data/half${name}A_${tvt}/idxmap.txt
    mv reduced_B.txt        data/half${name}B_${tvt}/files.txt
    mv reduced_B_idxmap.txt data/half${name}B_${tvt}/idxmap.txt
}

make_half_from_files_tvt ()
{
    make_half_from_files "$1" "$2" "$3" train
    make_half_from_files "$1" "$2" "$3" valid
    make_half_from_files "$1" "$2" "$3" test
}

make_reduced ()
{
    number="$1"
    tvt="$2"
    ./make_reduced_dataset.py --perclass "$number" data/whole_${tvt}/files.txt
    mkdir -p data/reduced${number}_${tvt}
    mv reduced.txt        data/reduced${number}_${tvt}/files.txt
}

# Random split datasets
#make_half_tvt 0
#make_half_tvt 1
#make_half_tvt 2
#make_half_tvt 3
#make_half_tvt 4
#make_half_tvt 5
#make_half_tvt 6
#make_half_tvt 7
#make_half_tvt 8
#make_half_tvt 9

# Natural vs. Manmade split datasets
#make_half_from_files_tvt data/half_nat_idx.dat data/half_man_idx.dat natman

# Reduced size Datasets
#make_reduced 1000 train
#make_reduced 0750 train
#make_reduced 0500 train
#make_reduced 0250 train
#make_reduced 0100 train
#make_reduced 0050 train
#make_reduced 0025 train
#make_reduced 0010 train
#make_reduced 0005 train
#make_reduced 0002 train
#make_reduced 0001 train
