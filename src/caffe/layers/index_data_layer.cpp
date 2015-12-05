#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
IndexDataLayer<Dtype>::~IndexDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template<typename Dtype>
void IndexDataLayer<Dtype>::CreatePrefetchThread() {
  StartInternalThread();
}

template<typename Dtype>
void IndexDataLayer<Dtype>::JoinPrefetchThread() {
  StopInternalThread();
}

template<typename Dtype>
void IndexDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  if (top.size() == MinTopBlobs()) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }

  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.index_data_param().backend()));
  db_->Open(this->layer_param_.index_data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Read a data point, and use it to initialize the top blob.
  IndexDatum datum;
  datum.ParseFromString(cursor_->value());
  height_data = datum.height_data();
  window_size = datum.size();

  this->prefetch_data_.reset(new IntegerBlob<Dtype>());
  this->prefetch_data_copy_.reset(new IntegerBlob<Dtype>());
  this->prefetch_label_.reset(new Blob<Dtype>());
  this->prefetch_label_copy_.reset(new Blob<Dtype>());

  vector<int> shape_vec(3);
  shape_vec[0] = this->layer_param_.index_data_param().batch_size();
  shape_vec[1] = height_data;
  shape_vec[2] = window_size;

  top[0]->Reshape(shape_vec);
  top[1]->Reshape(shape_vec);

  this->prefetch_data_->Reshape(shape_vec);
  this->prefetch_data_copy_->Reshape(shape_vec);
  LOG(INFO)<< "output data size: " << shape_vec[0] << ","
  << shape_vec[1] << "," << shape_vec[2];
  // label
  if (this->output_labels_) {
    vector<int> shape_label(1,
               this->layer_param_.index_data_param().batch_size());
    top[2]->Reshape(shape_label);
    this->prefetch_label_->Reshape(shape_label);
    this->prefetch_label_copy_->Reshape(shape_label);
  }

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  DLOG(INFO)<< "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO)<< "Prefetch initialized.";
}

// This function is used to create a thread that prefetches the data.
template<typename Dtype>
void IndexDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(prefetch_data_->count());

  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  int* int_data = prefetch_data_->mutable_cpu_indices();

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_->mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.index_data_param().batch_size();
  const int dtype_size = height_data * window_size;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    IndexDatum datum;
    datum.ParseFromString(cursor_->value());

    Dtype* destination = top_data + item_id * dtype_size;
    for (int k = 0; k < dtype_size; k++) {
      destination[k] = datum.data(k);
    }
    int * int_destination = int_data + item_id * window_size;
    for (int k = 0; k < window_size; k++) {
      int_destination[k] = datum.indices(k);
    }

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }

    // go to the next iter
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO)<< "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
}

template<typename Dtype>
void IndexDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // we swap the prefetch data
  prefetch_data_.swap(prefetch_data_copy_);
  prefetch_label_.swap(prefetch_label_copy_);

  // Start a new prefetch thread ahead of any memory transfer
  CreatePrefetchThread();

  top[0]->ShareData(*prefetch_data_copy_.get());
  top[1]->ShareData(*prefetch_data_copy_.get());

  if (output_labels_) {
    caffe_copy(prefetch_label_copy_->count(), prefetch_label_copy_->cpu_data(),
               top[2]->mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(IndexDataLayer, Forward);
#endif

INSTANTIATE_CLASS(IndexDataLayer);
REGISTER_LAYER_CLASS(IndexData);

}  // namespace caffe
