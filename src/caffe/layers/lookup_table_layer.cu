#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/integer_blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LookupTableLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  IntegerBlob<Dtype> * bottomIntegerBlob =
            dynamic_cast<IntegerBlob<Dtype>*>(bottom[0]);
    if (bottomIntegerBlob == 0) {
      LOG(FATAL)<< "the bottom blob is not an instance of IntegerBlob";
    }
    const int* bottom_data = bottomIntegerBlob->cpu_indices();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    caffe_gpu_scal(NUM_ * INPUT_SIZE_ * SIZE_, (Dtype) 0.0, top_data);

    for (int i = 0; i < NUM_; i++) {
      for (int w = 0; w < INPUT_SIZE_; w++) {
        const int pos = bottom_data[i * INPUT_SIZE_ + w];
        CHECK_GE(pos, 0);
        CHECK_LT(pos, N_INDEX_);

        caffe_gpu_axpy(SIZE_, (Dtype) 1.,
                       weight + pos * SIZE_,
                       top_data +(i * (INPUT_SIZE_ * SIZE_) + w),
                       1, INPUT_SIZE_);
      }
    }
}

template <typename Dtype>
void LookupTableLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();

      IntegerBlob<Dtype> * bottomIntegerBlob =
          dynamic_cast<IntegerBlob<Dtype>*>(bottom[0]);
      if (bottomIntegerBlob == 0) {
        LOG(FATAL)<< "the bottom blob is not an instance of IntegerBlob";
      }
      const int* bottom_data = bottomIntegerBlob->cpu_indices();
      Dtype* diff = this->blobs_[0]->mutable_gpu_diff();
      caffe_gpu_scal(N_INDEX_ * SIZE_, (Dtype) 0.0, diff);
      for (int i = 0; i < NUM_; i++) {
        for (int w = 0; w < INPUT_SIZE_; w++) {
          const int pos = bottom_data[i * INPUT_SIZE_ + w];
          CHECK_GE(pos, 0);
          CHECK_LT(pos, N_INDEX_);
          caffe_gpu_axpy(SIZE_, (Dtype) 1.,
                         top_diff +(i * (INPUT_SIZE_ * SIZE_) + w),
                         diff + pos * SIZE_, INPUT_SIZE_, 1);
        }
      }
    }
    if (propagate_down[0]) {
      LOG(FATAL)<< "The LookupTableLayer cannot propagate down";
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(LookupTableLayer);

}  // namespace caffe
