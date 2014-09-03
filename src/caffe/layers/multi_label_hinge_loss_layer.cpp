#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype MultiLabelHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "MultiLabelHingeLossLayer::Forward_cpu begin";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();

  CHECK_EQ(bottom[0]->num(),   bottom[1]->num())   << "data.num and label.num must match.";
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "data.count and label.count must match.";
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  // LOG(INFO) << "label dimension is " << dim;

  caffe_copy(count, bottom_data, bottom_diff);

  // OLD HingeLoss way
  //for (int i = 0; i < num; ++i) {
  //  bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  //}

  // Slow way??
  int this_label;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      this_label = label[i * dim + j];
      if (this_label == -1) {
        bottom_diff[i * dim + j] = std::max(Dtype(0), 1 + bottom_diff[i * dim + j]);
      } else if (this_label == 1) {
        bottom_diff[i * dim + j] = std::max(Dtype(0), 1 - bottom_diff[i * dim + j]);
      } else if (this_label == 0) {
        bottom_diff[i * dim + j] = Dtype(0);
      } else {
        LOG(ERROR) << "Label shoudl be -1, 0, or 1 but it is " << this_label;
      }
    }
  }

  Dtype loss;
  switch (this->layer_param_.multi_label_hinge_loss_param().norm()) {
  case MultiLabelHingeLossParameter_Norm_L1:
    loss = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case MultiLabelHingeLossParameter_Norm_L2:
    loss = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }

  LOG(INFO) << "MultiLabelHingeLossLayer::Forward_cpu end";
  LOG(INFO) << "Returning loss " << loss;
  return loss;
}

template <typename Dtype>
void MultiLabelHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  LOG(INFO) << "MultiLabelHingeLossLayer::Backward_cpu begin";
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();
    int dim = count / num;

    // OLD HingeLoss way
    // for (int i = 0; i < num; ++i) {
    //   bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    // }

    int this_label;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        this_label = label[i * dim + j];
        if (this_label == -1) {
          // Do Nothing
        } else if (this_label == 1) {
          bottom_diff[i * dim + j] *= -1;
        } else if (this_label == 0) {
          CHECK_EQ(bottom_diff[i * dim + j], Dtype(0)) << "This should already be 0 from the forward pass";
        } else {
          LOG(ERROR) << "Label shoudl be -1, 0, or 1 but it is " << this_label;
        }
      }
    }

    switch (this->layer_param_.multi_label_hinge_loss_param().norm()) {
    case MultiLabelHingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, Dtype(1. / num), bottom_diff);
      break;
    case MultiLabelHingeLossParameter_Norm_L2:
      caffe_scal(count, Dtype(2. / num), bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
  LOG(INFO) << "MultiLabelHingeLossLayer::Backward_cpu end";
}

INSTANTIATE_CLASS(MultiLabelHingeLossLayer);

}  // namespace caffe
