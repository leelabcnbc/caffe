#! /usr/bin/env python

# JBY

from wand.image import Image as WImage
from IPython.display import display



#im = WImage(filename='caffe/docs/caffe-presentation.pdf[13]')
#display(WImage(filename='caffe/docs/caffe-presentation.pdf[13]'))
#display(WImage(filename='caffe/docs/caffe-presentation.pdf[14]'))
def showpdf(filename, page=None, width=None):
    path = filename + ('' if page is None else '[%d]' % page)
    im = WImage(filename=path)
    if width is not None:
        cur = im.width, im.height
        im.resize(width, im.height * width / im.width)
    display(im)
