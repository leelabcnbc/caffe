"""
Wrap the internal caffe C++ module (_caffe.so) with a clean, Pythonic
interface.
"""

from ._caffe import CaffeNet
from collections import OrderedDict
import numpy as np

class Net(CaffeNet):
    """
    The direct Python interface to caffe, exposing Forward and Backward
    passes, data, gradients, and layer parameters
    """
    def __init__(self, param_file, pretrained_param_file):
        super(Net, self).__init__(param_file, pretrained_param_file)
        self._blobs = OrderedDict([(bl.name, bl)
                                   for bl in super(Net, self).blobs])
        self.params = OrderedDict([(lr.name, lr.blobs)
                                   for lr in super(Net, self).layers
                                   if len(lr.blobs) > 0])

    @property
    def blobs(self):
        """
        An OrderedDict (bottom to top, i.e., input to output) of network
        blobs indexed by name
        """
        return self._blobs

    @property
    def complete_layers(self):
        """
        A list of layers from bottom to top, including the data layer
        """
        return ['data'] + [lr.name for lr in self.layers]

    def ForwardFrom(self, input_layer, output_layer, input_data, shape_ref=None):
        """
        Set the layer with name input_layer to input_data, do a
        forward pass to the layer with name output_layer, and return
        the output of that layer. input_data must be the correct
        shape.
        """

        input_idx = self.complete_layers.index(input_layer)
        output_idx = self.complete_layers.index(output_layer)

        #input_blob = np.zeros(self.blobs[input_layer].data.shape, dtype=np.float32)
        if shape_ref == None:
            shape_ref = output_layer
        try:
            out_blob = self.blobs[shape_ref]
        except KeyError:
            raise Exception('Cannot figure out the output shape from layer '
                            '%s. Instead, provide a shape_ref that exists in'
                            ' .blobs, i.e. one of these: %s)' % (shape_ref, self.blobs))
        output_blob = np.zeros(out_blob.data.shape, dtype=np.float32)

        self.ForwardPartial([input_data], [output_blob], input_idx, output_idx)

        return output_blob

    def BackwardFrom(self, input_layer, output_layer, input_data):
        """
        Set the layer with name input_layer to input_data, do a
        backward pass to the layer with name output_layer, and return
        the diff at that output of that layer. input_data must be the correct
        shape.
        """

        input_idx = self.complete_layers.index(input_layer)
        output_idx = self.complete_layers.index(output_layer)

        shape_ref = output_layer
        try:
            out_blob = self.blobs[shape_ref]
        except KeyError:
            raise Exception('Cannot figure out the output shape from layer '
                            '%s. Instead, modify this function and provide a '
                            'shape_ref that exists in'
                            ' .blobs, i.e. one of these: %s)' % (shape_ref, self.blobs))
        output_blob = np.zeros(out_blob.data.shape, dtype=np.float32)

        #print '***p ', 'input_idx', input_idx, 'output_idx', output_idx
        self.BackwardPartial([input_data], [output_blob], input_idx, output_idx)

        return output_blob

    def deconv(self, input_layer, output_layer, input_data):
        '''Performs deconvolution through multiple layers'''

        #import ipdb as pdb; pdb.set_trace()
        
        input_idx = self.complete_layers.index(input_layer)
        output_idx = self.complete_layers.index(output_layer)

        cur_data = input_data
        for layer_idx in range(input_idx, output_idx, -1):
            cur_data = self.deconv_single(layer_idx, cur_data)
        return cur_data

    def deconv_single(self, layer_idx, input_data):
        '''Performs deconvolution through a single layer.'''

        assert input_data.dtype == np.float32, 'Must give float32 data, but got %s' % repr(input_data.dtype)
        assert layer_idx > 0, 'cannot deconv starting at data layer or below'
        
        #import ipdb as pdb; pdb.set_trace()
        
        input_idx = layer_idx       # e.g. 1 for conv1
        output_idx = layer_idx - 1  # e.g. 0 for data
        input_layer_name = self.complete_layers[input_idx]
        input_layer = self.layers[input_idx - 1]    # idx not considering data layer        
        output_layer_name = self.complete_layers[output_idx]

        print 'deconv_single from %d to %d (%s to %s)' % (input_idx, output_idx, input_layer_name, output_layer_name)

        if 'conv' in input_layer_name:
            #ret = self.BackwardFrom(layer_idx, layer_idx-1, input_data)
            output_blob = np.zeros(self.blobs[output_layer_name].data.shape, dtype=np.float32)

            #print '***p ', 'input_idx', input_idx, 'output_idx', output_idx
            self.BackwardPartial([input_data], [output_blob], input_idx, output_idx)

            return output_blob
        elif 'relu' in input_layer_name:
            return maximum(input_data, 0.0)
        elif 'pool' in input_layer_name:
            return maximum(input_data, 0.0)
        elif 'norm' in input_layer_name:
            print 'Warning: passing', input_layer_name, 'straight through'
            return input_data
        else:
            raise Exception('Not sure what to do with layer: %s' % layer_name)
        print 'done'
        return ret
