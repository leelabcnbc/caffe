#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <stdint.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(1073741824, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, cv_read_flag);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}





////////////////////////////////////
std::vector<hsize_t> hdf5_load_nd_dataset_helper_0(hid_t file_id, const char* dataset_name_, int min_dim, int max_dim, H5T_class_t expected_class, unsigned index_start, unsigned n_max, unsigned & index_max) {
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);
  CHECK_LE(max_dim, 4) << "Blobs only support up to 4 dims";
  CHECK_GE(index_start, 0);
  CHECK_GE(n_max, 0);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> data_dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
    file_id, dataset_name_, data_dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_LT(index_start, data_dims[0]) << "Cannot start at index " << index_start << " when num examples is " << data_dims[0];
  CHECK_EQ(class_, expected_class) << "Expected data of class " << expected_class << " but got class " << class_;

  std::vector<hsize_t> out_dims(data_dims);
  if (n_max == 0) {
    // We'll read from index_start until the end of the dataset
    out_dims[0] = data_dims[0] - index_start;
    index_max = data_dims[0];
  } else {
    // We'll read from index_start until either the end or n_max
    index_max = (data_dims[0] < index_start + n_max) ? data_dims[0] : index_start + n_max;
    out_dims[0] = index_max - index_start;
  }

  return out_dims;
}


template <typename Dtype>
void hdf5_load_nd_dataset_helper_1(Dtype* buffer, hid_t file_id, const char* dataset_name_, std::vector<hsize_t> out_dims, H5T_class_t expected_class, hid_t data_type, unsigned index_start, unsigned index_max) {
  hid_t dataset = H5Dopen(file_id, dataset_name_, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  
  int ndims = out_dims.size();
  hsize_t* offset = new hsize_t[ndims]; /* hyperslab offset in the file */
  hsize_t* count = new hsize_t[ndims];    /* size of the hyperslab in the file */
  offset[0] = index_start;  // slice block of the appropriate size from dim 0
  count[0] = index_max - index_start;
  for (int d = 1; d < ndims; d++) {
    // return the complete block of data along all dimensions except maybe 0
    offset[d] = 0;
    count[d] = out_dims[d];
  }
  // Select the hyperslab to be read
  herr_t status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
  delete[] offset;
  delete[] count;
  CHECK_GE(status, 0) << "Failed to select hyperslab for dataset " << dataset_name_;

  // Read the data   (INT SPECIFIC)
  status = H5Dread(dataset, data_type, H5S_ALL, dataspace,
                          H5P_DEFAULT, buffer);
  //status = H5LTread_dataset(
  //  file_id, dataset_name_, H5T_NATIVE_UCHAR, blob->mutable_cpu_data());
  H5Dclose(dataset);
  H5Sclose(dataspace);
  CHECK_GE(status, 0) << "Failed to read dataset " << dataset_name_;
}
////////////////////////////////////



// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob, H5T_class_t expected_class, hid_t data_type, unsigned index_start, unsigned n_max) {
  unsigned index_max;
  std::vector<hsize_t> out_dims = hdf5_load_nd_dataset_helper_0(file_id, dataset_name_, min_dim, max_dim, expected_class, index_start, n_max, index_max);

  blob->Reshape(
    out_dims[0],
    (out_dims.size() > 1) ? out_dims[1] : 1,
    (out_dims.size() > 2) ? out_dims[2] : 1,
    (out_dims.size() > 3) ? out_dims[3] : 1);

  hdf5_load_nd_dataset_helper_1(blob->mutable_cpu_data(), file_id, dataset_name_, out_dims, expected_class, data_type, index_start, index_max);
}


// TODO: Move to hpp file.
// hdf5_load_nd_dataset allows you to specify index_start for which
// row to start on and n_max for how many to read at most. 0 to start
// at beginning, 0 to read all rows.
template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob, unsigned index_start, unsigned n_max) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim,
                              blob, H5T_FLOAT, H5T_NATIVE_FLOAT, index_start, n_max);
}
template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset(file_id, dataset_name_, min_dim, max_dim, blob, 0, 0);
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob, unsigned index_start, unsigned n_max) {
  // Specify index_start for which row to start on and n_max for how many to read at most. 0 to start at beginning, 0 to read all rows.
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim,
                              blob, H5T_FLOAT, H5T_NATIVE_DOUBLE, index_start, n_max);
}
template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset(file_id, dataset_name_, min_dim, max_dim, blob, 0, 0);
}

// template <>
// void hdf5_load_nd_dataset<uint8_t>(hid_t file_id, const char* dataset_name_,
//         int min_dim, int max_dim, Blob<uint8_t>* blob, unsigned index_start, unsigned n_max) {
//   // Specify index_start for which row to start on and n_max for how many to read at most. 0 to start at beginning, 0 to read all rows.
//   hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim,
//                               blob, H5T_INTEGER, H5T_NATIVE_UCHAR, index_start, n_max);
// }
// template <>
// void hdf5_load_nd_dataset<uint8_t>(hid_t file_id, const char* dataset_name_,
//         int min_dim, int max_dim, Blob<uint8_t>* blob) {
//   hdf5_load_nd_dataset(file_id, dataset_name_, min_dim, max_dim, blob, 0, 0);
// }

// OLD
// template <>
// void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
//         int min_dim, int max_dim, Blob<float>* blob) {
//   H5T_class_t class_ = hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
//   CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";
//   herr_t status = H5LTread_dataset_float(
//     file_id, dataset_name_, blob->mutable_cpu_data());
//   CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
// }

// template <>
// void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
//         int min_dim, int max_dim, Blob<double>* blob) {
//   H5T_class_t class_ = hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
//   CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";
//   herr_t status = H5LTread_dataset_double(
//     file_id, dataset_name_, blob->mutable_cpu_data());
//   CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
//}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    const hid_t file_id, const string dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}  // namespace caffe
