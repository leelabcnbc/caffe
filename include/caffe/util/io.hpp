#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <string>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

#define HDF5_NUM_DIMS 4

namespace caffe {

using ::google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}


H5T_class_t hdf5_get_dataset_class(hid_t file_id, const char* dataset_name_);
std::vector<hsize_t> hdf5_get_dataset_datadims(hid_t file_id, const char* dataset_name_);
std::vector<hsize_t> hdf5_get_dataset_datumdims(hid_t file_id, const char* dataset_name_);
hsize_t hdf5_get_dataset_num_examples(hid_t file_id, const char* dataset_name_);

std::vector<hsize_t> hdf5_load_nd_dataset_helper_0(hid_t file_id, const char* dataset_name_, int min_dim, int max_dim, H5T_class_t expected_class, unsigned index_start, unsigned n_max, unsigned & index_max);

template <typename Dtype>
void hdf5_load_nd_dataset_helper_1(Dtype* buffer, hid_t file_id, const char* dataset_name_, std::vector<hsize_t> datum_dims, hid_t data_type, unsigned index_start, unsigned index_max);

template <typename Dtype>
void hdf5_load_nd_dataset_helper(
  hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
  Blob<Dtype>* blob, H5T_class_t expected_class, hid_t data_type, unsigned index_start, unsigned n_max);
// OLD
//template <typename Dtype>
//void hdf5_load_nd_dataset_helper(
//hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
//Blob<Dtype>* blob, unsigned n_start, unsigned n_max);

// hdf5_load_nd_dataset allows you to specify index_start for which
// row to start on and n_max for how many to read at most. 0 to start
// at beginning, 0 to read all rows.

template <typename Dtype>
void hdf5_load_nd_dataset(
  hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
  Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_nd_dataset(
  hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
  Blob<Dtype>* blob, unsigned index_start, unsigned n_max);

template <typename Dtype>
unsigned hdf5__jason_load(hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
                          Dtype** ptr_buffer, unsigned buffer_examples_offset,
                          unsigned batch_size,
                          unsigned index_start, unsigned n_max) {
  unsigned index_max;
  std::vector<hsize_t> out_dims = hdf5_load_nd_dataset_helper_0(
    file_id, dataset_name_, min_dim, max_dim, (H5T_class_t) NULL,
    index_start, n_max, index_max);

  std::vector<hsize_t> datum_dims = hdf5_get_dataset_datumdims(file_id, dataset_name_);

  CHECK_LE(buffer_examples_offset, batch_size) << "offset into buffer should be less than total batch size";
  CHECK_GE(batch_size - buffer_examples_offset, index_max - index_start)
    << "Remaining space in buffer is smaller than batch about to be read";

  int datum_count = 1;
  for (int i = 0; i < datum_dims.size(); ++i)
    datum_count *= datum_dims[i];

  // Allocate buffer for whole batch_size if it is not already allocated
  LOG(INFO) << "Note: this is data layer for file_id " << file_id;
  LOG(INFO) << "Before check, buffer address is: " << *ptr_buffer;
  if (!*ptr_buffer) {
    LOG(INFO) << "Allocing more memory, number of elems: " << batch_size * datum_count;
    *ptr_buffer = new Dtype[batch_size * datum_count];
  }
  LOG(INFO) << "After check, buffer address is: " << *ptr_buffer;

  Dtype* buffer_partition_start = *ptr_buffer + buffer_examples_offset * datum_count;
  hid_t output_type = get_hdf5_type(buffer_partition_start);
  LOG(INFO) << "Loading dataset to buffer starting at " << buffer_partition_start;
  hdf5_load_nd_dataset_helper_1(buffer_partition_start, file_id, dataset_name_, datum_dims,
    output_type, index_start, index_max);
  return index_max - index_start;
}


template <typename Dtype>
void hdf5_save_nd_dataset(
  const hid_t file_id, const string dataset_name, const Blob<Dtype>& blob);

template <typename Dtype>
hid_t get_hdf5_type(const Dtype* junk);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
