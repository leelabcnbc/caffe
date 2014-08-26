#include <iostream>
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
//#include "io.hpp"

using namespace std;

/*  
 *   This example reads hyperslab from the SDS.h5 file 
 *   created by h5_write.c program into two-dimensional
 *   plane of the three-dimensional array. 
 *   Information about dataset in the SDS.h5 file is obtained. 
 */
 
#define FLOATFILE  "test_float.h5"
#define UINTFILE   "test_uint.h5"
#define DATASETNAME "data"

//#define NX_SUB  3           /* hyperslab dimensions */ 
//#define NY_SUB  4 

#define NB 2
#define NC 3
#define NI 4           /* output buffer dimensions */ 
#define NJ 5
#define N_total (NB * NC * NI * NJ)
#define N_slab ((SLABEND-SLABSTART) * NC * NI * NJ)

#define SLABSTART 1
#define SLABEND 2

#define NDIM 4




//////////////////////////////////


std::vector<hsize_t> hdf5_get_dataset_datadims(hid_t file_id, const char* dataset_name_) {
  /* Gets a vector of the dataset dimensions (n_examples, dim_0, dim_1, ...) */
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);

  std::vector<hsize_t> data_dims(ndims);
  status = H5LTget_dataset_info(file_id, dataset_name_, data_dims.data(), NULL, NULL);

  return data_dims;
}

std::vector<hsize_t> hdf5_get_dataset_datumdims(hid_t file_id, const char* dataset_name_) {
  /* Gets a vector of the datum dimensions (dim_0, dim_1, ...) */
  std::vector<hsize_t> data_dims = hdf5_get_dataset_datadims(file_id, dataset_name_);
  // Remove the first dimension, which is the number of examples in the dataset
  data_dims.erase(data_dims.begin());
  return data_dims;
}

template < class T >
inline std::ostream& operator << (std::ostream& os, const std::vector<T>& v) 
{
  os << "[";
  for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
    {
      os << " " << *ii;
    }
  os << " ]";
  return os;
 }

//////////////////////////////////



void make_dataset_float()
{
  hid_t       file, dataset;         /* file and dataset handles */
  hid_t       datatype, dataspace;   /* handles */
  herr_t      status;                             
  //float       data[NX][NY];          /* data to write */

  float* buffer = new float[N_total];

  /* 
   * Data  and output buffer initialization. 
   */
  int counter = 0;
  for (int b = 0; b < NB; b++) {
    for (int c = 0; c < NC; c++) {
      for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
          buffer[counter] = counter;
          counter++;
        }
      }
    }
  }
  /*
   * 0 1 2 3 4 5 
   * 1 2 3 4 5 6
   * 2 3 4 5 6 7
   * 3 4 5 6 7 8
   * 4 5 6 7 8 9
   */

  /*
   * Create a new file using H5F_ACC_TRUNC access,
   * default file creation properties, and default file
   * access properties.
   */
  file = H5Fcreate(FLOATFILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  /*
   * Describe the size of the array and create the data space for fixed
   * size dataset. 
   */
  hsize_t     dimsf[4];              /* dataset dimensions */
  dimsf[0] = NB;
  dimsf[1] = NC;
  dimsf[2] = NI;
  dimsf[3] = NJ;
  dataspace = H5Screate_simple(4, dimsf, NULL);

  /* 
   * Define datatype for the data in the file.
   * We will store little endian INT numbers.
   */
  datatype = H5Tcopy(H5T_NATIVE_FLOAT);
  //status = H5Tset_order(datatype, H5T_ORDER_LE);

  /*
   * Create a new dataset within the file using defined dataspace and
   * datatype and default dataset creation properties.
   */
  /*
hid_t H5Dcreate2(hid_t, const char*, hid_t, hid_t, hid_t, hid_t, hid_t)
  */
  dataset = H5Dcreate(file, DATASETNAME, datatype, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /*
   * Write the data to the dataset using default transfer properties.
   */
  status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, buffer);

  /*
   * Close/release resources.
   */
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Dclose(dataset);
  H5Fclose(file);
}

void make_dataset_uint8()
{
  hid_t       file, dataset;         /* file and dataset handles */
  hid_t       datatype, dataspace;   /* handles */
  herr_t      status;                             
  //float       data[NX][NY];          /* data to write */

  char* buffer = new char[N_total];

  /* 
   * Data  and output buffer initialization. 
   */
  int counter = 0;
  for (int b = 0; b < NB; b++) {
    for (int c = 0; c < NC; c++) {
      for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
          buffer[counter] = counter;
          counter++;
        }
      }
    }
  }
  /*
   * 0 1 2 3 4 5 
   * 1 2 3 4 5 6
   * 2 3 4 5 6 7
   * 3 4 5 6 7 8
   * 4 5 6 7 8 9
   */

  /*
   * Create a new file using H5F_ACC_TRUNC access,
   * default file creation properties, and default file
   * access properties.
   */
  file = H5Fcreate(UINTFILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  /*
   * Describe the size of the array and create the data space for fixed
   * size dataset. 
   */
  hsize_t     dimsf[4];              /* dataset dimensions */
  dimsf[0] = NB;
  dimsf[1] = NC;
  dimsf[2] = NI;
  dimsf[3] = NJ;
  dataspace = H5Screate_simple(4, dimsf, NULL);

  /* 
   * Define datatype for the data in the file.
   * We will store little endian INT numbers.
   */
  datatype = H5Tcopy(H5T_NATIVE_UINT8);
  //status = H5Tset_order(datatype, H5T_ORDER_LE);

  /*
   * Create a new dataset within the file using defined dataspace and
   * datatype and default dataset creation properties.
   */
  /*
hid_t H5Dcreate2(hid_t, const char*, hid_t, hid_t, hid_t, hid_t, hid_t)
  */
  dataset = H5Dcreate(file, DATASETNAME, datatype, dataspace,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /*
   * Write the data to the dataset using default transfer properties.
   */
  status = H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, buffer);

  /*
   * Close/release resources.
   */
  H5Sclose(dataspace);
  H5Tclose(datatype);
  H5Dclose(dataset);
  H5Fclose(file);
}


// void read_dataset()
// {
//   hid_t       file, dataset;         /* handles */
//   hid_t       datatype, dataspace;   
//   hid_t       memspace; 
//   H5T_class_t dtype_class;                 /* datatype class */
//   H5T_order_t order;                 /* data order */
//   size_t      size;                  /*
//                                       * size of the data element       
//                                       * stored in file
//                                       */
//   hsize_t     dimsm[3];              /* memory space dimensions */
//   hsize_t     dims_out[2];           /* dataset dimensions */      
//   herr_t      status;                             

//   int         data_out[NB][NC][NI][NJ]; /* output buffer */
   
//   hsize_t      count[2];              /* size of the hyperslab in the file */
//   hsize_t      offset[2];             /* hyperslab offset in the file */
//   hsize_t      count_out[3];          /* size of the hyperslab in memory */
//   hsize_t      offset_out[3];         /* hyperslab offset in memory */
//   int          i, j, k, status_n, rank;

//   for (j = 0; j < NX; j++) {
//     for (i = 0; i < NY; i++) {
//       for (k = 0; k < NZ ; k++)
//         data_out[j][i][k] = 0;
//     }
//   } 
 
//   /*
//    * Open the file and the dataset.
//    */
//   file = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
//   dataset = H5Dopen(file, DATASETNAME, H5P_DEFAULT);

//   /*
//    * Get datatype and dataspace handles and then query
//    * dataset class, order, size, rank and dimensions.
//    */
//   datatype  = H5Dget_type(dataset);     /* datatype handle */ 
//   dtype_class     = H5Tget_class(datatype);
//   if (dtype_class == H5T_INTEGER) printf("Data set has INTEGER type \n");
//   order     = H5Tget_order(datatype);
//   if (order == H5T_ORDER_LE) printf("Little endian order \n");

//   size  = H5Tget_size(datatype);
//   printf(" Data size is %d \n", size);

//   dataspace = H5Dget_space(dataset);    /* dataspace handle */
//   rank      = H5Sget_simple_extent_ndims(dataspace);
//   status_n  = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
//   printf("rank %d, dimensions %lu x %lu \n", rank,
//          (unsigned long)(dims_out[0]), (unsigned long)(dims_out[1]));

//   /* 
//    * Define hyperslab in the dataset. 
//    */
//   offset[0] = 1;
//   offset[1] = 2;
//   count[0]  = NX_SUB;
//   count[1]  = NY_SUB;
//   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, 
//                                count, NULL);

//   /*
//    * Define the memory dataspace.
//    */
//   dimsm[0] = NX;
//   dimsm[1] = NY;
//   dimsm[2] = NZ ;
//   memspace = H5Screate_simple(RANK_OUT,dimsm,NULL);   

//   /* 
//    * Define memory hyperslab. 
//    */
//   offset_out[0] = 3;
//   offset_out[1] = 0;
//   offset_out[2] = 0;
//   count_out[0]  = NX_SUB;
//   count_out[1]  = NY_SUB;
//   count_out[2]  = 1;
//   status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, 
//                                count_out, NULL);

//   /*
//    * Read data from hyperslab in the file into the hyperslab in 
//    * memory and display.
//    */
//   status = H5Dread(dataset, H5T_NATIVE_INT, memspace, dataspace,
//                    H5P_DEFAULT, data_out);
//   for (j = 0; j < NX; j++) {
//     for (i = 0; i < NY; i++) printf("%d ", data_out[j][i][0]);
//     printf("\n");
//   }
//   /*
//    * 0 0 0 0 0 0 0
//    * 0 0 0 0 0 0 0
//    * 0 0 0 0 0 0 0
//    * 3 4 5 6 0 0 0  
//    * 4 5 6 7 0 0 0
//    * 5 6 7 8 0 0 0
//    * 0 0 0 0 0 0 0
//    */

//   /*
//    * Close/release resources.
//    */
//   H5Tclose(datatype);
//   H5Dclose(dataset);
//   H5Sclose(dataspace);
//   H5Sclose(memspace);
//   H5Fclose(file);
// }     


void read_dataset_jason_float()
{
  hid_t file_id = H5Fopen(FLOATFILE, H5F_ACC_RDONLY, H5P_DEFAULT);

  std::vector<hsize_t> data_dims = hdf5_get_dataset_datadims(file_id, "data");
  cout << "Loaded data with dims " << data_dims << endl;

  float* buffer = new float[N_total];
  cout << "Allocated buffer of size " << N_total << " at location " << buffer << endl;
  cout << "Loading dataset to buffer starting at " << buffer << endl;

  hid_t output_type = H5T_NATIVE_FLOAT;
  cout << "output_type is " << output_type << " with size " << H5Tget_size(output_type) << endl;

  hid_t dataset = H5Dopen(file_id, DATASETNAME, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
    
  H5Dread(dataset, output_type, H5S_ALL, dataspace, H5P_DEFAULT, buffer);

  cout << "Read this data:" << endl;
  int counter = 0;
  for (int b = 0; b < NB; b++) {
    for (int c = 0; c < NC; c++) {
      for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
          cout << buffer[counter] << " ";
          counter++;
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
  }

  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Fclose(file_id);
}

void read_dataset_jason_float_slab()
{
  // INPUT SELECTIONS
  hid_t file_id = H5Fopen(FLOATFILE, H5F_ACC_RDONLY, H5P_DEFAULT);

  std::vector<hsize_t> data_dims = hdf5_get_dataset_datadims(file_id, "data");
  cout << "Loaded data with dims " << data_dims << endl;

  hid_t output_type = H5T_NATIVE_FLOAT;
  cout << "output_type is " << output_type << " with size " << H5Tget_size(output_type) << endl;
  
  hid_t dataset = H5Dopen(file_id, DATASETNAME, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);


  hsize_t* offset = new hsize_t[NDIM]; /* hyperslab offset in the file */
  hsize_t* count = new hsize_t[NDIM];    /* size of the hyperslab in the file */
  offset[0] = SLABSTART;
  count[0] = SLABEND - SLABSTART;
  for (int d = 1; d < NDIM; d++) {
    // return the complete block of data along all dimensions except maybe 0
    offset[d] = 0;
    count[d] = data_dims[d];
  }

  cout << "offset is [";
  for (int d = 0; d < NDIM; d++)
    cout << " " << offset[d];
  cout << " ]" << endl;
  cout << "count is [";
  for (int d = 0; d < NDIM; d++)
    cout << " " << count[d];
  cout << " ]" << endl;

  // Select the hyperslab to be read
  cout << "dataspace is " << dataspace << endl;
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
  cout << "dataspace is " << dataspace << endl;
  delete[] offset;
  delete[] count;
  //CHECK_GE(status, 0) << "Failed to select hyperslab for dataset " << dataset_name_;


  // OUTPUT SELECTIONS
  hsize_t* dims_buffer = new hsize_t[NDIM];    /* size of the hyperslab in the file */
  dims_buffer[0] = SLABEND - SLABSTART;
  dims_buffer[1] = NC;
  dims_buffer[2] = NI;
  dims_buffer[3] = NJ;
  hid_t memspace = H5Screate_simple(NDIM, dims_buffer, NULL);

  cout << "dims_buffer is [";
  for (int d = 0; d < NDIM; d++)
    cout << " " << dims_buffer[d];
  cout << " ]" << endl;

  /* 
   * Define memory hyperslab. 
   */
  // Skip for now??
  // offset_out[0] = 3;
  // offset_out[1] = 0;
  // offset_out[2] = 0;
  // count_out[0]  = NX_SUB;
  // count_out[1]  = NY_SUB;
  // count_out[2]  = 1;
  // status = H5Sselect_hyperslab (memspace, H5S_SELECT_SET, offset_out, NULL, 
  //                               count_out, NULL);

  float* buffer = new float[N_slab];
  cout << "Allocated buffer of size " << N_slab << " at location " << buffer << endl;
  cout << "Loading dataset to buffer starting at " << buffer << endl;

  // WORKS!
  // First line fails, second works! Makes no sense given the most straightforward interpretation of the documentation
  //H5Dread(dataset, output_type, H5S_ALL, dataspace, H5P_DEFAULT, buffer);
  H5Dread(dataset, output_type, memspace, dataspace, H5P_DEFAULT, buffer);

  cout << "Read this data:" << endl;
  int counter = 0;
  for (int b = 0; b < (SLABEND-SLABSTART); b++) {
    for (int c = 0; c < NC; c++) {
      for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
          cout << buffer[counter] << " ";
          counter++;
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
  }

  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Fclose(file_id);
}

void read_dataset_jason_int2float_slab()
{
  // INPUT SELECTIONS
  hid_t file_id = H5Fopen(UINTFILE, H5F_ACC_RDONLY, H5P_DEFAULT);

  std::vector<hsize_t> data_dims = hdf5_get_dataset_datadims(file_id, "data");
  cout << "Loaded data with dims " << data_dims << endl;
  
  hid_t dataset = H5Dopen(file_id, DATASETNAME, H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);


  hsize_t* offset = new hsize_t[NDIM]; /* hyperslab offset in the file */
  hsize_t* count = new hsize_t[NDIM];    /* size of the hyperslab in the file */
  offset[0] = SLABSTART;
  count[0] = SLABEND - SLABSTART;
  for (int d = 1; d < NDIM; d++) {
    // return the complete block of data along all dimensions except maybe 0
    offset[d] = 0;
    count[d] = data_dims[d];
  }

  cout << "offset is [";
  for (int d = 0; d < NDIM; d++)
    cout << " " << offset[d];
  cout << " ]" << endl;
  cout << "count is [";
  for (int d = 0; d < NDIM; d++)
    cout << " " << count[d];
  cout << " ]" << endl;

  // Select the hyperslab to be read
  cout << "dataspace is " << dataspace << endl;
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
  cout << "dataspace is " << dataspace << endl;
  delete[] offset;
  delete[] count;
  //CHECK_GE(status, 0) << "Failed to select hyperslab for dataset " << dataset_name_;


  // OUTPUT SELECTIONS
  hid_t output_type = H5T_NATIVE_FLOAT;
  cout << "output_type is " << output_type << " with size " << H5Tget_size(output_type) << endl;

  hsize_t* dims_buffer = new hsize_t[NDIM];    /* size of the hyperslab in the file */
  dims_buffer[0] = SLABEND - SLABSTART;
  dims_buffer[1] = NC;
  dims_buffer[2] = NI;
  dims_buffer[3] = NJ;
  hid_t memspace = H5Screate_simple(NDIM, dims_buffer, NULL);

  cout << "dims_buffer is [";
  for (int d = 0; d < NDIM; d++)
    cout << " " << dims_buffer[d];
  cout << " ]" << endl;

  /* 
   * Define memory hyperslab. 
   */
  // Skip for now??
  // offset_out[0] = 3;
  // offset_out[1] = 0;
  // offset_out[2] = 0;
  // count_out[0]  = NX_SUB;
  // count_out[1]  = NY_SUB;
  // count_out[2]  = 1;
  // status = H5Sselect_hyperslab (memspace, H5S_SELECT_SET, offset_out, NULL, 
  //                               count_out, NULL);

  float* buffer = new float[N_slab];
  cout << "Allocated buffer of size " << N_slab << " at location " << buffer << endl;
  cout << "Loading dataset to buffer starting at " << buffer << endl;

  // WORKS!
  // First line fails, second works! Makes no sense given the most straightforward interpretation of the documentation
  //H5Dread(dataset, output_type, H5S_ALL, dataspace, H5P_DEFAULT, buffer);
  H5Dread(dataset, output_type, memspace, dataspace, H5P_DEFAULT, buffer);

  cout << "Read this data:" << endl;
  int counter = 0;
  for (int b = 0; b < (SLABEND-SLABSTART); b++) {
    for (int c = 0; c < NC; c++) {
      for (int i = 0; i < NI; i++) {
        for (int j = 0; j < NJ; j++) {
          cout << buffer[counter] << " ";
          counter++;
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
  }

  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Fclose(file_id);
}

// void read_dataset_jason()
// {
//   std::vector<hsize_t> hdf_data_datumdims_;  // does not include batch size
//   std::vector<hsize_t> hdf_label_datumdims_;  // does not include batch size

//   float* hdf_buffer_data_ = NULL; // For partial reads and reads of uncropped images
//   float* hdf_buffer_label_ = NULL; // For partial reads
//   unsigned hdf_buffer_loaded_ = 0; // How many data points have been loaded to buffer, // How much of buffer_data_ and buffer_label_ have been filled

//   // NEW General
//   int label_channels_ = -1;

//   int datum_channels_ = -1;
//   int datum_height_ = -1;  // pre-crop height and width
//   int datum_width_ = -1;
//   int datum_size_ = -1;


//   hid_t file_id = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);

//   hdf_data_datumdims_ = hdf5_get_dataset_datumdims(file_id, "data");
//   cout << "Loaded hdf_data_datumdims_ of size " << hdf_data_datumdims_.size() << endl;

//   cout << "Before read, buffer is " << hdf_buffer_data_;
//   //n_data_read = hdf5__jason_load(file_id, "data", MIN_DATA_DIM, MAX_DATA_DIM,
//   //                               hdf_buffer_data_, hdf_buffer_loaded_, batch_size,
//   //                               hdf_current_row_, batch_size - hdf_buffer_loaded_);
//   //  hdf5__jason_load vvvvvvv

//   unsigned index_max;

//   //  hdf5_load_nd_dataset_helper_0 vvvvvvv
//   //std::vector<hsize_t> out_dims = hdf5_load_nd_dataset_helper_0(
//   //  file_id, dataset_name_, min_dim, max_dim, (H5T_class_t) NULL,
//   //  index_start, n_max, index_max);
//   // Some sanity checks

//   std::vector<hsize_t> data_dims = hdf5_get_dataset_datadims(file_id, DATASETNAME);
//   // Verify that the number of dimensions is in the accepted range.
//   hsize_t ndims = data_dims.size();
//   cout << "ndims is " << ndims << endl;
//   // If given a non-zero class, verify that the data format is the one we expet

//   std::vector<hsize_t> out_dims(data_dims);
//   if (n_max == 0) {
//     // We'll read from index_start until the end of the dataset
//     out_dims[0] = data_dims[0] - index_start;
//     index_max = data_dims[0];
//   } else {
//     // We'll read from index_start until either the end or n_max
//     index_max = (data_dims[0] < index_start + n_max) ? data_dims[0] : index_start + n_max;
//     out_dims[0] = index_max - index_start;
//   }

//   cout << "out_dims is " << out_dims << endl;

//   //  hdf5_load_nd_dataset_helper_0 ^^^^^^^

//   std::vector<hsize_t> datum_dims = hdf5_get_dataset_datumdims(file_id, dataset_name_);

//   cout << "datum_dims is " << datum_dims << endl;

//   int datum_count = 1;
//   for (int i = 0; i < datum_dims.size(); ++i)
//     datum_count *= datum_dims[i];

//   cout << "datum_count is " << datum_count << endl;

//   buffer = hdf_buffer_data_;

//   // Allocate buffer for whole batch_size if it is not already allocated
//   //HERE_MOVE_THIS_OUTSIDE;
//   cout << "Note: this is data layer for file_id " << file_id;
//   cout << "Before check, buffer address is: " << buffer;
//   if (!buffer) {
//     cout << "Allocing more memory, number of elems: " << batch_size * datum_count
//               << " size of Dtype " << sizeof(Dtype)
//               << " = total bytes " << batch_size * datum_count * sizeof(Dtype);
//     buffer = new Dtype[batch_size * datum_count];
//   } else {
//     cout << "Not allocing more memory, number of elems should be: " << batch_size * datum_count;
//   }
//   cout << "After check, buffer address is: " << buffer;

//   Dtype* buffer_partition_start = buffer + buffer_examples_offset * datum_count;
//   hid_t output_type = get_hdf5_type(buffer_partition_start);
//   cout << "Loading dataset to buffer starting at " << buffer_partition_start;
//   cout << "output_type is " << output_type << " with size " << H5Tget_size(output_type);
//   hdf5_load_nd_dataset_helper_1(buffer_partition_start, file_id, dataset_name_, datum_dims,
//     output_type, index_start, index_max);
//   cout << "Just after load_nd_helper_1";

//   cout << "Loading dataset to buffer starting at " << buffer_partition_start;
//   cout << "output_type is " << output_type << " with size " << H5Tget_size(output_type);
//   hdf5_load_nd_dataset_helper_1(buffer_partition_start, file_id, dataset_name_, datum_dims,
//     output_type, index_start, index_max);
//   cout << "Just after load_nd_helper_1";

//   return index_max - index_start;















//     //  hdf5__jason_load ^^^^^^^
//     cout << "After read, buffer is " << hdf_buffer_data_;









//     herr_t status = H5Fclose(file_id);
//     CHECK_GE(status, 0) << "Failed to close HDF5 file " << filename;

// }

int main()
{
  //make_dataset_float();
  //make_dataset_uint8();
  //read_dataset();
  
  //read_dataset_jason_float();
  //read_dataset_jason_float_slab();
  read_dataset_jason_int2float_slab();

  return 0;
}
