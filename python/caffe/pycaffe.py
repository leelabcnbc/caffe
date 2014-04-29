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
                            '%s. Instead, modify this functino and provide a '
                            'shape_ref that exists in'
                            ' .blobs, i.e. one of these: %s)' % (shape_ref, self.blobs))
        output_blob = np.zeros(out_blob.data.shape, dtype=np.float32)

        #print '***p ', 'input_idx', input_idx, 'output_idx', output_idx
        self.BackwardPartial([input_data], [output_blob], input_idx, output_idx)

        return output_blob
