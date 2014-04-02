#! /usr/bin/env python

import argparse
from numpy import *




def makeit(old_idx):
    set_old_idx = set(old_idx)
    new_idx = range(len(old_idx))   # New is 0...max
    map_old_to_new = dict(zip(old_idx, new_idx))
    map_new_to_old = dict(zip(new_idx, old_idx))

    read_file = open(filename)
    write_file = open(write_filename, 'w')

    for line in read_file:
        jpg_file, old_class = line.split()
        old_class = int(old_class)
        if old_class in set_old_idx:
            new_class = map_new_to_old[old_class]
            write_file.write('%s %d' % (jpg_file, new_class))



def main():
    parser = argparse.ArgumentParser(description='Creates reduced datasets.')
    #parser.add_argument('--show', action = 'store_true',
    #                    help = 'Show plots as well (default: off)')
    parser.add_argument('--seed', type = int, nargs=1, default=[0],
                        help = 'Which seed to use (default: 0)')
    args = parser.parse_args()


    print 'seed is', args.seed[0]

    random.seed(args.seed)


    
    # FIRST: A/B split!
    idx = arange(1000)   # class indices
    groupAOldIdx = sorted(random.choice(idx, 500, replace=False))
    groupBOldIdx = sorted(list(set(idx)-set(groupA)))

    makeit(groupAOldIdx)
    



if __name__ == '__main__':
    main()
