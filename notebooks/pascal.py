#! /usr/bin/env python

# Functions for the Pascal VOC dataset

from pylab import figure, imshow, histogram, zeros


class_name_bg = 'background'
class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

n_class = len(class_names)
n_class_plus = n_class + 1   # one more class to include background



def conf_counts_single(gt_seg, proposed_seg, verbose = False):
    '''This is a Python version of the important parts of VOCevalseg.m.
    Returns conf_counts, the 21x21 confusion matrix of pixel labels'''
    
    assert proposed_seg.min() >= 0,  'Proposal includes index that is too small'
    assert proposed_seg.max() <= n_class, 'Proposal includes index that is too large'
    assert proposed_seg.shape == gt_seg.shape, 'Proposal is wrong size (%s instead of %s)' % (proposed_seg.shape, gt_seg.shape)

    # Matlab: locs = gtim<255;
    non_void = gt_seg.flatten() < 255   # which pixel locations should be counted, i.e. everything except the void (yellow) regions
    
    if verbose:
        print 'Ground truth labels in this image are:', sorted(set(gt_seg.flatten()))
    
        print 'Pixels in image         ', len(gt_seg.flatten())
        print 'Non-void pixels in image', len(gt_seg.flatten()[non_void])
    
    # Matlab: sumim = 1+gtim+resim*num;
    sum_im = gt_seg + proposed_seg * n_class_plus
    if verbose:
        _=imshow(sum_im)
    
    # Matlab: hs = histc(sumim(locs),1:num*num);
    conf_counts,bins = histogram(sum_im.flatten()[non_void], bins=range(n_class_plus**2+1))
    
    if verbose:
        # Compute confusion matrix
        # Matlab:
        # % confusion matrix - first index is true label, second is inferred label
        # conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
        # Slower version with tiled matrix
        # conf_mat = 100 * conf_counts_all / tile(reshape(conf_counts_all.sum(1), (conf_counts_all.shape[0],1)), (1,conf_counts_all.shape[1]))
        # Faster version with broadcasting
        conf_mat = 100 * ((conf_counts_all).T / (conf_counts_all.sum(1) + 1e-20)).T
        figure()
        _=imshow(conf_mat)
    
    return conf_counts

def accuracies_from_conf(conf_counts_all, verbose = False):
    # Matlab:
    # accuracies = zeros(VOCopts.nclasses,1);
    # fprintf('Accuracy for each class (intersection/union measure)\n');
    # for j=1:num
    #
    #   gtj=sum(confcounts(j,:));
    #   resj=sum(confcounts(:,j));
    #   gtjresj=confcounts(j,j);
    #   % The accuracy is: true positive / (true positive + false positive + false negative)
    #   % which is equivalent to the following percentage:
    #   accuracies(j)=100*gtjresj/(gtj+resj-gtjresj);
    #
    #   clname = 'background';
    #   if (j>1), clname = VOCopts.classes{j-1};end;
    #   fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
    # end
    # accuracies = accuracies(1:end);
    # avacc = mean(accuracies);
    # fprintf('-------------------------\n');
    # fprintf('Average accuracy: %6.3f%%\n',avacc);
    
    accuracies = zeros(n_class_plus)
    for class_id in range(n_class_plus):
        gtj     = conf_counts_all[class_id,:].sum()
        resj    = conf_counts_all[:,class_id].sum()
        gtjresj = conf_counts_all[class_id,class_id]
        accuracies[class_id] = 100 * gtjresj/(gtj+resj-gtjresj)
        
        if verbose:
            clname = class_name_bg
            if class_id > 0:
                clname = class_names[class_id-1]
            print '%2d: %15s, %g' % (class_id, clname, accuracies[class_id])

    return accuracies

def DEP_conf_counts_all():
    '''Expects an iterator or list of (name, proposed_seg) tuples.'''
    # Matlab: confcounts = zeros(num);
    conf_counts_all = zeros((n_class_plus,n_class_plus))

    for ii in range(10):
        conf_counts = accuracies_single(gt_seg, proposed_seg)

        # Matlab: confcounts(:) = confcounts(:) + hs(:);
        conf_counts_all.flat += conf_counts

