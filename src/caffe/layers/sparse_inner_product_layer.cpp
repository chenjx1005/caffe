#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/sparse_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SparseInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // three bottom blobs are: value, indices and ptr
  M_ = bottom[2]->count() - 1;
  K_ = this->layer_param_.sparse_inner_product_param().input_dim();
  N_ = this->layer_param_.sparse_inner_product_param().num_output();
  bias_term_ = this->layer_param_.sparse_inner_product_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
  	if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.sparse_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.sparse_inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void SparseInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // The top shape will M_ * N_
  vector<int> top_shape(2, M_);
  top_shape[1] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void SparseInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* value = bottom[0]->cpu_data();
  const Dtype* indices = bottom[1]->cpu_data();
  const Dtype* ptr = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_csrgemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_, (Dtype) 1.,
      value, indices, ptr, weight, (Dtype) 0., top_data);

  if (this->bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
      	this->bias_multiplier_.cpu_data(), 
      	this->blobs_[1]->cpu_data(), (Dtype) 1., top_data);
  }
}

template <typename Dtype>
void SparseInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
      // Gradient with respect to weight
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* value = bottom[0]->cpu_data();
      const Dtype* indices = bottom[1]->cpu_data();
      const Dtype* ptr = bottom[2]->cpu_data();
      caffe_cpu_csrgemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype) 1., value, indices, ptr, top_diff,
          (Dtype) 1., this->blobs_[0]->mutable_cpu_diff());
    }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // there is a bug in the code because this is called no matter what!
    LOG(INFO) << "propagate down not supported for sparse inner product";
  }
}

#ifdef CPU_ONLY
STUB_GPU(SparseInnerProductLayer);
#endif

INSTANTIATE_CLASS(SparseInnerProductLayer);
REGISTER_LAYER_CLASS(SparseInnerProduct);

}  // namespace caffe
