#! /usr/bin/env python

# JBY

from IPython.display import display



#im = WImage(filename='caffe/docs/caffe-presentation.pdf[13]')
#display(WImage(filename='caffe/docs/caffe-presentation.pdf[13]'))
#display(WImage(filename='caffe/docs/caffe-presentation.pdf[14]'))
def showpdf(filename, page=None, width=None):
    from wand.image import Image as WImage
    path = filename + ('' if page is None else '[%d]' % page)
    im = WImage(filename=path)
    if width is not None:
        cur = im.width, im.height
        im.resize(width, im.height * width / im.width)
    display(im)


# Some function definitions
def figsize(width,height):
    from pylab import rcParams
    rcParams['figure.figsize'] = (width,height)


class DuckStruct(object):
    '''Use to store anything!'''

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        rep = ['%s=%s' % (k, repr(v)) for k,v in self.__dict__.items()]
        return 'DuckStruct(%s)' % ', '.join(rep)
