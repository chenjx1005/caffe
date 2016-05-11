#ifndef CAFFE_IMAGE_SAMPLE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SAMPLE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageSampleDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSampleDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageSampleDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSampleData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages(int i);
  virtual void ShuffleClass();
  virtual void load_batch(Batch<Dtype>* batch);

  int class_num_;
  vector<vector<std::pair<std::string, int> > > lines_;
  vector<int> lines_id_;
  int class_id_;
  vector<int> lines_size_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
