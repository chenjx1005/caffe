#include <functional>
#include <utility>
#include <vector>

#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void BiClassifyAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  threshold_ = this->layer_param_.biclassify_accuracy_param().threshold();
  postive_ = this->layer_param_.biclassify_accuracy_param().postive();
  negative_ = this->layer_param_.biclassify_accuracy_param().negative();
}

template <typename Dtype>
void BiClassifyAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions; ";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void BiClassifyAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int count = 0;
  for (int i = 0; i < bottom[0]->shape(0); ++i) {
    const int label_value = static_cast<int>(bottom_label[i]);

    CHECK(label_value==postive_||label_value==negative_);
    const int predict_value = bottom_data[i] > threshold_ ? postive_ : negative_;
    if (label_value == predict_value) accuracy++;
    ++count;
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
}

INSTANTIATE_CLASS(BiClassifyAccuracyLayer);
REGISTER_LAYER_CLASS(BiClassifyAccuracy);

}  // namespace caffe
