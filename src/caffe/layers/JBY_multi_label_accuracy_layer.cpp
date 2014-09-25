#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void JBY_MultiLabelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "The data and label should have the same count.";
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
Dtype JBY_MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype correct = 0;
  Dtype seen = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int this_proposed, this_actual;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      this_actual = bottom_label[i * dim + j];
      if (this_actual == 0) {
        // Ignore classifications when true label is 0 (ambiguous)
        continue;
      } else if (this_actual == 1) {
        correct += bottom_data > 0 ? 1 : 0;
      } else if (this_actual == -1) {
        correct += bottom_data < 0 ? 1 : 0;
      } else {
        LOG(ERROR) << "Expected actual label of -1, 0, or 1 but got " << this_actual;
      }
      seen++;
    }
  }

  accuracy = correct / (seen + Dtype(1e-12)) / (num+dim);
  LOG(INFO) << "Computed average accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy;

  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(JBY_MultiLabelAccuracyLayer);

}  // namespace caffe
