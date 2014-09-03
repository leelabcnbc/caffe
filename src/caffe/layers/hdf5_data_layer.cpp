/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  LOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(ERROR) << "Failed opening HDF5 file" << filename;
    return;
  }

  // EVENTUALLY: HERE
  // Outside: "What do we do with n_max?? if n_max < remaining rows,
  // we'll be fine. But if n_max is greater, need to read partial and
  // then return read size."  "If (b), then make a new blob, of the
  // given size"

  const int MIN_DATA_DIM = 2;
  const int MAX_DATA_DIM = 4;
  hdf5_load_nd_dataset(
    file_id, "data",  MIN_DATA_DIM, MAX_DATA_DIM, &data_blob_);
  LOG(INFO) << "JBY: loaded data_blob_ with size (num: " << data_blob_.num()
            << ", channels: " << data_blob_.channels()
            << ", height: " << data_blob_.height()
            << ", width: " << data_blob_.width()
            << ", count: " << data_blob_.count() << ")";

  const int MIN_LABEL_DIM = 1;
  const int MAX_LABEL_DIM = 2;
  hdf5_load_nd_dataset(
    file_id, "label", MIN_LABEL_DIM, MAX_LABEL_DIM, &label_blob_);
  LOG(INFO) << "JBY: loaded label_blob_ with size (num: " << label_blob_.num()
            << ", channels: " << label_blob_.channels()
            << ", height: " << label_blob_.height()
            << ", width: " << label_blob_.width()
            << ", count: " << label_blob_.count() << ")";

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file " << filename;
  CHECK_EQ(data_blob_.num(), label_blob_.num());
  LOG(INFO) << "Successully loaded " << data_blob_.num() << " rows";
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading HDF5 filenames from " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of files: " << num_files_;

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  LOG(INFO) << "JBY: batch size is: " << batch_size;
  (*top)[0]->Reshape(batch_size, data_blob_.channels(),
                     data_blob_.width(), data_blob_.height());
  (*top)[1]->Reshape(batch_size, label_blob_.channels(),
                     label_blob_.width(), label_blob_.height());
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
}

template <typename Dtype>
Dtype HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();  // e.g. 10
  const int data_count = (*top)[0]->count() / (*top)[0]->num();              // JBY: size of one data point = 3*256*256 = 196608
  const int label_data_count = (*top)[1]->count() / (*top)[1]->num();        // JBY: size of one label = 1
  //I0815 16:08:43.662384 30319 hdf5_data_layer.cpp:40] JBY: loaded data_blob_ with size (num: 1000, channels: 3, height: 256, width: 256, count: 196608000)
  //I0815 16:08:43.662591 30319 hdf5_data_layer.cpp:50] JBY: loaded label_blob_ with size (num: 1000, channels: 1, height: 1, width: 1, count: 1000)

  LOG(INFO) << "JBY: HDF5DataLayer<Dtype>::Forward_cpu";
  for (int i = 0; i < batch_size; ++i, ++current_row_) {
    if (current_row_ == data_blob_.num()) {
      if (num_files_ > 1) {
        current_file_ += 1;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          LOG(INFO) << "looping around to first file";
        }
        LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
      }
      current_row_ = 0;
    }
    memcpy(&(*top)[0]->mutable_cpu_data()[i * data_count],
           &data_blob_.cpu_data()[current_row_ * data_count],
           sizeof(Dtype) * data_count);
    memcpy(&(*top)[1]->mutable_cpu_data()[i * label_data_count],
            &label_blob_.cpu_data()[current_row_ * label_data_count],
            sizeof(Dtype) * label_data_count);
  }
  return Dtype(0.);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);

}  // namespace caffe
