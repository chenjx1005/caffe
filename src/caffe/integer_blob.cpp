#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/integer_blob.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void IntegerBlob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_EQ(shape.size(), 3);
  Blob<Dtype>::Reshape(shape);
  if (this->count_ > 0) {
    LOG(INFO)<< "indices setup to" <<
    this->shape_[0] * this->shape_[2] << "\n\n";
    indices_.reset(new SyncedMemory(
        this->shape_[0] * this->shape_[2] * sizeof(int)));
  } else {
    LOG(INFO) << "indices setup to null\n";
    indices_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  }
}

template<typename Dtype>
void IntegerBlob<Dtype>::Reshape(const int num, const int channels,
                                const int height, const int width) {
  CHECK_EQ(width, 1);
  vector<int> shape(3);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;

  Reshape(shape);
}

template<typename Dtype>
IntegerBlob<Dtype>::IntegerBlob(const vector<int>& shape) {
  Reshape(shape);
}

template<typename Dtype>
IntegerBlob<Dtype>::IntegerBlob(const int num, const int channels,
                                const int height, const int width) {
  Reshape(num, channels, height, width);
}

template<typename Dtype>
const int* IntegerBlob<Dtype>::cpu_indices() const {
  CHECK(indices_);
  return (const int*) indices_->cpu_data();
}

template<typename Dtype>
const int* IntegerBlob<Dtype>::gpu_indices() const {
  CHECK(indices_);
  return (const int*) indices_->gpu_data();
}

template<typename Dtype>
int* IntegerBlob<Dtype>::mutable_cpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_cpu_data());
}

template<typename Dtype>
int* IntegerBlob<Dtype>::mutable_gpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_gpu_data());
}

template<typename Dtype>
void IntegerBlob<Dtype>::CopyFrom(const Blob<Dtype>& source, bool copy_diff,
                                 bool reshape) {
  LOG(FATAL)<< "CopyFrom is not supported";
}

template<typename Dtype>
void IntegerBlob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  LOG(FATAL)<< "FromProto is not supported";
}

template<typename Dtype>
void IntegerBlob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  LOG(FATAL)<< "ToProto is not supported";
}

template <typename Dtype>
void IntegerBlob<Dtype>::ShareData(const Blob<Dtype>& other) {
  CHECK_EQ(this->count_, other.count());
  const IntegerBlob<Dtype>* integerBlob =
        dynamic_cast<const IntegerBlob<Dtype>*>(&other);
  CHECK(integerBlob);

  this->data_ = integerBlob->data();
  indices_ = integerBlob->indices();
}

template<typename Dtype>
void IntegerBlob<Dtype>::ShareDiff(const Blob<Dtype>& other) {
  LOG(FATAL)<< "ShareDiff is not supported";
}


INSTANTIATE_CLASS(IntegerBlob);
}  // namespace caffe

