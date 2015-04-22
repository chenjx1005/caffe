#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/integer_blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void LookupTableLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  N_INDEX_ = this->layer_param_.lookup_table_param().n_index();
  SIZE_ = this->layer_param_.lookup_table_param().size();
  // Figure out the dimensions
  const vector<int> bottom_shape = bottom[0]->shape();
  CHECK_EQ(bottom_shape.size(), 3);
  NUM_ = bottom_shape[0];
  INPUT_SIZE_ = bottom_shape[2];

  LOG(INFO) << "input_size: " << INPUT_SIZE_ <<
      " num: " << NUM_ << " size: "<< SIZE_ <<
      " N_INDEX: " << N_INDEX_ << "\n";
  top[0]->Reshape(NUM_, SIZE_, INPUT_SIZE_, 1);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO)<< "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> par_shape(2);
    par_shape[0] = N_INDEX_;
    par_shape[1] = SIZE_;
    this->blobs_[0].reset(new Blob<Dtype>(par_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.lookup_table_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void LookupTableLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  IntegerBlob<Dtype> * bottomIntegerBlob =
          dynamic_cast<IntegerBlob<Dtype>*>(bottom[0]);
  if (bottomIntegerBlob == 0) {
    LOG(FATAL)<< "the bottom blob is not an instance of IntegerBlob";
  }
  const int* bottom_data = bottomIntegerBlob->cpu_indices();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_scal(NUM_ * INPUT_SIZE_ * SIZE_, (Dtype) 0.0, top_data);

  for (int i = 0; i < NUM_; i++) {
    for (int w = 0; w < INPUT_SIZE_; w++) {
      const int pos = bottom_data[i * INPUT_SIZE_ + w];
      CHECK_GE(pos, 0);
      CHECK_LT(pos, N_INDEX_);
      caffe_axpy(SIZE_, (Dtype) 1., weight + pos * SIZE_,
                 top_data +(i * (INPUT_SIZE_ * SIZE_) + w), 1, INPUT_SIZE_);
    }
  }
}

template<typename Dtype>
void LookupTableLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();

    IntegerBlob<Dtype> * bottomIntegerBlob =
        dynamic_cast<IntegerBlob<Dtype>*>(bottom[0]);
    if (bottomIntegerBlob == 0) {
      LOG(FATAL)<< "the bottom blob is not an instance of IntegerBlob";
    }
    const int* bottom_data = bottomIntegerBlob->cpu_indices();
    Dtype* diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_scal(N_INDEX_ * SIZE_, (Dtype) 0.0, diff);
    for (int i = 0; i < NUM_; i++) {
      for (int w = 0; w < INPUT_SIZE_; w++) {
        const int pos = bottom_data[i * INPUT_SIZE_ + w];
        CHECK_GE(pos, 0);
        CHECK_LT(pos, N_INDEX_);
        caffe_axpy(SIZE_, (Dtype) 1., top_diff +(i * (INPUT_SIZE_ * SIZE_) + w),
                   diff + pos * SIZE_, INPUT_SIZE_, 1);
      }
    }
  }
  if (propagate_down[0]) {
    LOG(FATAL)<< "The LookupTableLayer cannot propagate down";
  }
}

#ifdef CPU_ONLY
STUB_GPU(LookupTableLayer);
#endif

INSTANTIATE_CLASS(LookupTableLayer);
REGISTER_LAYER_CLASS(LookupTable);

}  // namespace caffe
