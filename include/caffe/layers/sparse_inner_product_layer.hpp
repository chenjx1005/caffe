#ifndef CAFFE_SPARSE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_SPARSE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief InnerProduct layer with sparse matrix as input. Compressed Sparse
 * 		  Row matrix (CSR) format is used to represent a batch of data. The
 *		  layer should be given 3 bottom blobs for indptr, indices and value.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
 template <typename Dtype>
class SparseInnerProductLayer : public Layer<Dtype> {
 public:
  explicit SparseInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SparseInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace caffe

#endif  // CAFFE_SPARSE_INNER_PRODUCT_LAYER_HPP_