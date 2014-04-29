## Install Log

This is a log of my experience setting up Caffe on Ubuntu 13.04 by following these two sets of instructions:

- http://caffe.berkeleyvision.org/installation.html
- https://github.com/BVLC/caffe/wiki/Installation-%28rbg%29

Notes:

- Here we use Ubuntu 13.04 instead of 12.04, as assumed by the second link above.
- When starting this installation, I already had Cuda 5.5 installed.
- I had a checkout of caffe at `$HOME/s/caffe`. If your checkout is at a different location, just change any instances of `s/caffe` to the correct path.



## Steps to build caffe

Try to compile:

    jason [~/s/caffe] $ make
    /usr/bin/g++ src/caffe/proto/caffe.pb.cc -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -c -o build/src/caffe/proto/caffe.pb.o
    In file included from src/caffe/proto/caffe.pb.cc:4:0:
    ./src/caffe/proto/caffe.pb.h:9:42: fatal error: google/protobuf/stubs/common.h: No such file or directory
    compilation terminated.
    make: *** [build/src/caffe/proto/caffe.pb.o] Error 1

Fails, so install libprotobuf-dev:

    jason [~] $ sudo apt-get install libprotobuf-dev

Try to compile:

    jason [~/s/caffe] $ make
    /usr/bin/g++ src/caffe/proto/caffe.pb.cc -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -c -o build/src/caffe/proto/caffe.pb.o
    
    /usr/bin/g++ src/caffe/layer_factory.cpp -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -c -o build/src/caffe/layer_factory.o
    In file included from ./include/caffe/blob.hpp:6:0,
                     from ./include/caffe/layer.hpp:7,
                     from src/caffe/layer_factory.cpp:8:
    ./include/caffe/common.hpp:6:32: fatal error: boost/shared_ptr.hpp: No such file or directory
    compilation terminated.
    make: *** [build/src/caffe/layer_factory.o] Error 1

Fails, so install boost:

    jason [~] $ sudo apt-get install libboost-all-dev

Note: this installed boost 1.49 by default on Ubuntu 13.04.

Try to compile:

    jason [~/s/caffe] $ make
    /usr/bin/g++ src/caffe/layer_factory.cpp -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -c -o build/src/caffe/layer_factory.o
    In file included from ./include/caffe/blob.hpp:6:0,
                     from ./include/caffe/layer.hpp:7,
                     from src/caffe/layer_factory.cpp:8:
    ./include/caffe/common.hpp:12:26: fatal error: glog/logging.h: No such file or directory

Fails, so install google-glog:

    jason [~/s/caffe] $ mkdir -p ~/temp/glog
    jason [~/s/caffe] $ cd ~/temp/glog
    jason [~/temp/glog] $ wget 'https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz'
    ...
    jason [~/temp/glog] $ tar xvzf glog-0.3.3.tar.gz
    ...
    jason [~/temp/glog] $ cd glog-0.3.3/
    jason [~/temp/glog/glog-0.3.3] $ ./configure --prefix=$HOME/s/caffe/local
    ...
    jason [~/temp/glog/glog-0.3.3] $ make
    ...
    jason [~/temp/glog/glog-0.3.3] $ make install
    ...
    ----------------------------------------------------------------------
    Libraries have been installed in:
       /home/jason/s/caffe/local/lib
    
    If you ever happen to want to link against installed libraries
    in a given directory, LIBDIR, you must either use libtool, and
    specify the full pathname of the library, or use the `-LLIBDIR'
    flag during linking and do at least one of the following:
       - add LIBDIR to the `LD_LIBRARY_PATH' environment variable
         during execution
       - add LIBDIR to the `LD_RUN_PATH' environment variable
         during linking
       - use the `-Wl,-rpath -Wl,LIBDIR' linker flag
       - have your system administrator add LIBDIR to `/etc/ld.so.conf'
    
    See any operating system documentation about shared libraries for
    more information, such as the ld(1) and ld.so(8) manual pages.
    ----------------------------------------------------------------------
    ...

Try to compile:

    jason [~/s/caffe] $ make
    /usr/bin/g++ src/caffe/layer_factory.cpp -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -c -o build/src/caffe/layer_factory.o
    In file included from ./include/caffe/blob.hpp:6:0,
                     from ./include/caffe/layer.hpp:7,
                     from src/caffe/layer_factory.cpp:8:
    ./include/caffe/common.hpp:12:26: fatal error: glog/logging.h: No such file or directory
    compilation terminated.
    make: *** [build/src/caffe/layer_factory.o] Error 1

Oops, it can't find the logging headers we just installed in `$HOME/s/caffe/local`. Let's tell it where to look and then try again. Add the lines reflected by this diff to the file `Makefile.config`:

    jason [~/s/caffe] $ diff -u Makefile.config.example Makefile.config
    --- Makefile.config.example    2014-02-19 18:23:18.287192652 -0500
    +++ Makefile.config            2014-02-25 13:36:17.966603821 -0500
    @@ -22,8 +22,8 @@
     PYTHON_LIB := /usr/local/lib
    
     # Whatever else you find you need goes here.
    -INCLUDE_DIRS := $(PYTHON_INCLUDES) /usr/local/include
    -LIBRARY_DIRS := $(PYTHON_LIB) /usr/lib /usr/local/lib
    +INCLUDE_DIRS := $(PYTHON_INCLUDES) /usr/local/include     ./local/include
    +LIBRARY_DIRS := $(PYTHON_LIB) /usr/lib /usr/local/lib     ./local/lib
    
     # For OS X, use clang++.

Try to compile:

    jason [~/s/caffe] $ make
    /usr/bin/g++ src/caffe/layer_factory.cpp -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -I./local/include -c -o build/src/caffe/layer_factory.o
    In file included from ./include/caffe/blob.hpp:6:0,
                     from ./include/caffe/layer.hpp:7,
                     from src/caffe/layer_factory.cpp:8:
    ./include/caffe/common.hpp:13:21: fatal error: mkl_vsl.h: No such file or directory
    compilation terminated.
    make: *** [build/src/caffe/layer_factory.o] Error 1

Install the Intel Math Kernel Library (MKL) from here: http://software.intel.com/en-us/intel-mkl . (Get evaluation version if desired). When installing, choose option 3 and install to the default local path, `$HOME/intel`.

Try to compile:

    jason [~/s/caffe] $ make
    /usr/bin/g++ src/caffe/layer_factory.cpp -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./src -I./include -I/usr/local/cuda/include -I/opt/intel/mkl/include -I./local/include -c -o build/src/caffe/layer_factory.o
    In file included from ./include/caffe/blob.hpp:6:0,
                     from ./include/caffe/layer.hpp:7,
                     from src/caffe/layer_factory.cpp:8:
    ./include/caffe/common.hpp:13:21: fatal error: mkl_vsl.h: No such file or directory
    compilation terminated.
    make: *** [build/src/caffe/layer_factory.o] Error 1

Oops, it can't find the MKL library we just installed. Tell it where to look by adding the lines reflected by this diff to the file `Makefile.config`:

    jason [~/s/caffe] $ diff -u Makefile.config.example Makefile.config
    --- Makefile.config.example    2014-02-19 18:23:18.287192652 -0500
    +++ Makefile.config            2014-02-25 13:36:17.966603821 -0500
    @@ -8,7 +8,7 @@
         -gencode arch=compute_35,code=sm_35
    
     # MKL directory contains include/ and lib/ directions that we need.
    -MKL_DIR := /opt/intel/mkl
    +MKL_DIR := $HOME/intel/composerxe/mkl
    
     # NOTE: this is required only if you will compile the matlab interface.
     # MATLAB directory should contain the mex binary in /bin

Try to compile:

    jason [~/s/caffe] $ make
    /usr/bin/g++ src/caffe/layer_factory.cpp -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./local/include -I./src -I./include -I/usr/local/cuda/include -I/home/jason/intel/composerxe/mkl/include -c -o build/src/caffe/layer_factory.o
    In file included from src/caffe/layer_factory.cpp:9:0:
    ./include/caffe/vision_layers.hpp:6:24: fatal error: leveldb/db.h: No such file or directory
    compilation terminated.
    make: *** [build/src/caffe/layer_factory.o] Error 1

Fails, so install leveldb:

    jason [~] $ sudo apt-get install libleveldb-dev

Try to compile:

    jason [~/s/caffe] $ make
    ....
    /usr/bin/g++ src/caffe/util/io.cpp -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./local/include -I./src -I./include -I/usr/local/cuda/include -I/home/jason/intel/composerxe/mkl/include -c -o build/src/caffe/util/io.o
    src/caffe/util/io.cpp:8:33: fatal error: opencv2/core/core.hpp: No such file or directory
    compilation terminated.
    make: *** [build/src/caffe/util/io.o] Error 1

Fails, so install OpenCV. Here we use version 2.4.8 from git. Change the `wget` line to another snapshot from https://github.com/Itseez/opencv/releases if desired!

    jason [~] $ mkdir -p ~/temp/opencv
    jason [~] $ cd ~/temp/opencv
    jason [~/temp/opencv] $ wget 'https://github.com/Itseez/opencv/archive/2.4.8.tar.gz'
    ...
    jason [~/temp/opencv] $ tar xvzf 2.4.8.tar.gz
    ...
    jason [~/temp/opencv] $ cd opencv-2.4.8/
    jason [~/temp/opencv/opencv-2.4.8] $ mkdir release
    jason [~/temp/opencv/opencv-2.4.8] $ cd release/
    jason [~/temp/opencv/opencv-2.4.8/release] $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$HOME/s/caffe/local -D BUILD_opencv_gpu=OFF ..
    ...
    jason [~/temp/opencv/opencv-2.4.8/release] $ make
    ... takes a while ...
    jason [~/temp/opencv/opencv-2.4.8/release] $ make install
    ...

Try to compile:

    jason [~/s/caffe] $ make
    .............
    /usr/bin/g++ -shared -o libcaffe.so build/src/caffe/proto/caffe.pb.o build/src/caffe/layer_factory.o build/src/caffe/layers/lrn_layer.o build/src/caffe/layers/pooling_layer.o build/src/caffe/layers/neuron_layer.o build/src/caffe/layers/data_layer.o build/src/caffe/layers/conv_layer.o build/src/caffe/layers/inner_product_layer.o build/src/caffe/layers/flatten_layer.o build/src/caffe/layers/im2col_layer.o build/src/caffe/util/math_functions.o build/src/caffe/util/im2col.o build/src/caffe/util/io.o build/src/caffe/blob.o build/src/caffe/common.o build/src/caffe/net.o build/src/caffe/syncedmem.o build/src/caffe/solver.o build/src/caffe/layers/relu_layer.cuo build/src/caffe/layers/padding_layer.cuo build/src/caffe/layers/lrn_layer.cuo build/src/caffe/layers/loss_layer.cuo build/src/caffe/layers/softmax_loss_layer.cuo build/src/caffe/layers/dropout_layer.cuo build/src/caffe/layers/sigmoid_layer.cuo build/src/caffe/layers/softmax_layer.cuo build/src/caffe/layers/pooling_layer.cuo build/src/caffe/layers/bnll_layer.cuo build/src/caffe/util/im2col.cuo build/src/caffe/util/math_functions.cuo -pthread -fPIC -DNDEBUG -O2 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/include -I./local/include -I./src -I./include -I/usr/local/cuda/include -I/home/jason/intel/composerxe/mkl/include -L/usr/local/lib -L/usr/lib -L/usr/local/lib -L./local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/home/jason/intel/composerxe/mkl/lib -L/home/jason/intel/composerxe/mkl/lib/intel64 -lcudart -lcublas -lcurand -lmkl_rt -lpthread -lglog -lprotobuf -lleveldb -lsnappy -lboost_system -lopencv_core -lopencv_highgui -lopencv_imgproc -Wall
    /usr/bin/ld: cannot find -lsnappy
    collect2: error: ld returned 1 exit status
    make: *** [libcaffe.so] Error 1

Fails, so install snappy:

    jason [~/temp/opencv/opencv-2.4.8/release] $ sudo apt-get install libsnappy-dev

Try to compile:

    jason [~/s/caffe] $ make

It works!





## Bonus: build pycaffe

After installing all the prereqs above, this worked the first time for me:

    jason [~/s/caffe] $ make pycaffe

It works!





## MNIST example

Running the MNIST example from http://caffe.berkeleyvision.org/mnist.html is straightforward, but that page was missing a step, so I've included a log below:


    jason [~/s/caffe/data] $ ./get_mnist.sh
    Downloading...
    Unzipping...
    Done.

We also need to run `./create_mnist.sh` to create the leveldb files from the `*-ubyte` files we just downloaded:
    
    jason [~/s/caffe/data] $ ./create_mnist.sh
    Creating leveldb...
    ../build/examples/convert_mnist_data.bin: error while loading shared libraries: libglog.so.0: cannot open shared object file: No such file or directory
    ../build/examples/convert_mnist_data.bin: error while loading shared libraries: libglog.so.0: cannot open shared object file: No such file or directory
    Done.

Oops, this doesn't work because it's not sure where to find `libglog`. We'd better add `$HOME/s/caffe/local/lib` to our `LD_LIBRARY_PATH`. Let's add the MKL library location as well (this line is for intel64):
    
    jason [~/s/caffe/data] $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/s/caffe/local/lib:$HOME/intel/composerxe/mkl/lib/intel64
    
Now creation of the leveldb files should work:

    jason [~/s/caffe/data] $ ./create_mnist.sh
    Creating leveldb...
    Done.

    jason@kronk [~/s/caffe/data] $ ls -d *leveldb
    mnist-test-leveldb/  mnist-train-leveldb/

Now we can train the net! Note that my binary ended up in `../build/examples/train_net.bin` instead of at `../examples/train_net.bin`:

    jason@kronk [~/s/caffe/data] $ GLOG_logtostderr=1 ../build/examples/train_net.bin lenet_solver.prototxt
    I0225 17:14:17.583492 20761 train_net.cpp:26] Starting Optimization
    I0225 17:14:17.583827 20761 solver.cpp:26] Creating training net.
    I0225 17:14:17.583868 20761 net.cpp:66] Creating Layer mnist
    I0225 17:14:17.583884 20761 net.cpp:101] mnist -> data
    ...





## Notes

In case it's useful for debugging, below are for reference almost all the environment variables I have defined.

    TERM=xterm
    SHELL=/bin/bash
    SSH_CLIENT=127.0.0.1 60259 22
    OPENBLAS_NUM_THREADS=4
    USER=jason
    LD_LIBRARY_PATH=:/usr/local/cuda-5.5/lib64:/lib:/home/jason/s/caffe/local/lib:/home/jason/intel/composerxe/mkl/lib/intel64
    VIRTUAL_ENV=/home/jason/virtualenvs/ml
    PATH=/home/jason/virtualenvs/ml/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-5.5/bin
    LANG=en_US.UTF-8
    SHLVL=1
    HOME=/home/jason
    PYTHONPATH=/home/jason/local/python:/home/jason/s/pylearn2
    CUDA_ROOT=/usr/local/cuda-5.5

