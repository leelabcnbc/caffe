#! /usr/bin/env python

import argparse
from numpy import *
from matplotlib.pyplot import plot, subplot, figure, semilogy, legend, savefig, show, hold, axis, title, xlabel, ylabel



def insert_most_recent(dictionary, key, value, timestamp):
    if key in dictionary:
        stored_item = dictionary[key]
        stored_value, stored_timestamp = stored_item
        if timestamp > stored_timestamp:
            dictionary[key] = (value, timestamp)
    else:
        dictionary[key] = (value, timestamp)



def read_info(filenames):
    loss_train = {}
    lr = {}
    test_score_0 = {}
    test_score_1 = {}
    for filename in filenames:
        expecting_test_lines = 0
        ff = open(filename)
        for line in ff:
            # We need to deal with lines like the following:
            #   I0311 23:51:37.270727 24248 solver.cpp:207] Iteration 230020, lr = 0.0001
            #   I0311 23:51:37.433928 24248 solver.cpp:65] Iteration 230020, loss = 1.56229
            # 
            #   I0311 14:41:38.035086 19955 solver.cpp:87] Iteration 236000, Testing net
            #   I0311 14:43:52.531891 19955 solver.cpp:114] Test score #0: 0.56976
            #   I0311 14:43:52.535956 19955 solver.cpp:114] Test score #1: 1.83616
            fields = line.split()

            if len(fields) <= 4:
                continue

            timestamp = fields[0] + ' ' + fields[1]

            if expecting_test_lines > 0:
                # We're one of the lines just after the 'Testing net' line
                assert 'Test score #' in line, 'Unexpected pattern found'
                if 'Test score #0:' in line:
                    insert_most_recent(test_score_0, testing_iter, float(fields[7]), timestamp)
                elif 'Test score #1:' in line:
                    insert_most_recent(test_score_1, testing_iter, float(fields[7]), timestamp)
                else:
                    raise Exception('Expected test score 0 or 1.')
                expecting_test_lines -= 1
                if expecting_test_lines == 0:
                    testing_iter = None
                    
            elif fields[4] == 'Iteration':
                iteration = int(fields[5].strip(','))
                if fields[6] == 'lr':
                    insert_most_recent(lr, iteration, float(fields[8]), timestamp)
                elif fields[6] == 'loss':
                    insert_most_recent(loss_train, iteration, float(fields[8]), timestamp)
                elif ' '.join(fields[6:8]) == 'Testing net':
                    testing_iter = iteration
                    expecting_test_lines = 2
                            
    return loss_train, lr, test_score_0, test_score_1



def convert_dict(dd):
    keys = array(sorted(dd.keys()))
    #if len(unique(diff(keys))) != 1:
    #    raise Exception('Expected keys to be evenly spaced, but they are not.')
    values = []
    for key in keys:
        values.append(dd[key][0])
    return keys, array(values)





def plot_perf(logfiles):
    dicts = read_info(logfiles)

    arrays = [convert_dict(dd) for dd in dicts]
    #loss_train, lr, test_score_0, test_score_1 = dicts

    loss_train_it, loss_train = arrays[0]
    lr_it, lr = arrays[1]
    ts0_it, ts0 = arrays[2]
    ts1_it, ts1 = arrays[3]

    val_top1_err_it = ts0_it
    val_top1_err    = 1-ts0



    ###### plot_normal
    figure()

    pl = plot
    #pl = semilogy

    subplot(3,1,1)
    hold(True)
    every = 10
    pl(loss_train_it[::every], loss_train[::every], 'b-')
    pl(ts1_it, ts1, 'r-')
    axis('tight')
    ylabel('Cost')

    subplot(3,1,2)
    plot(val_top1_err_it, val_top1_err, 'g-')
    ylabel('Top-1 error')
    
    subplot(3,1,3)
    semilogy(lr_it, lr)
    ylabel('Learning Rate')
    xlabel('Iteration')

    savefig('plot_normal.pdf')



    ###### plot_zoom
    figure()

    subplot(2,1,1)
    hold(True)
    every = 10
    pl(loss_train_it[::every], loss_train[::every], 'b-')
    pl(ts1_it, ts1, 'r-')
    axis('tight')
    ylabel('Cost')
    
    axis('tight')
    ax = axis()
    frac = .05
    yrange = [percentile(ts1, 0), percentile(ts1, 60)]
    yrange = ((1+frac) * yrange[0] - frac * yrange[1], (1+frac) * yrange[1] - frac * yrange[0])
    axis(ax[0:2] + yrange)

    subplot(2,1,2)
    plot(val_top1_err_it, val_top1_err, 'g-')
    ylabel('Top-1 error')
    xlabel('Iteration')

    axis('tight')
    ax = axis()
    frac = .05
    yrange = [percentile(val_top1_err, 0), percentile(val_top1_err, 60)]
    yrange = ((1+frac) * yrange[0] - frac * yrange[1], (1+frac) * yrange[1] - frac * yrange[0])
    axis(ax[0:2] + yrange)

    savefig('plot_zoom.pdf')



def main():
    parser = argparse.ArgumentParser(description='Plots performance given one or more logfiles.')
    parser.add_argument('--show', action = 'store_true',
                        help = 'Show plots as well (default: off)')
    parser.add_argument('logfiles', type = str, nargs='+',
                        help = 'Log file(s) to scrape for performance numbers')
    args = parser.parse_args()

    plot_perf(args.logfiles)
    
    if args.show:
        show()
        from IPython import embed
        embed()



if __name__ == '__main__':
    main()
