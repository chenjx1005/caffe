#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
TxtDataLayer<Dtype>::~TxtDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TxtDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = 1;
  const int new_width  = 1;
  length_ = this->layer_param_.txt_data_param().length();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.txt_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string group_key;
  string noclick_click;
  string sparse_data;
  string column_name;
  while (infile >> group_key >> noclick_click >> sparse_data >> column_name) {
    lines_.push_back(std::make_pair(noclick_click, sparse_data));
  }

  if (this->layer_param_.txt_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLines();
  }
  LOG(INFO) << "A total of " << lines_.size() << " lines.";

  lines_id_ = 0;
  vector<int> top_shape(4, 1);
  top_shape[1] = length_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.txt_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void TxtDataLayer<Dtype>::ShuffleLines() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void TxtDataLayer<Dtype>::ParseLines(std::pair<std::string, std::string> line, Blob<Dtype>& data_blob, Dtype& label)
{
    vector<string> strs;
    boost::split(strs, line.first, boost::is_any_of(","));
    Dtype* data = data_blob.mutable_cpu_data();
    for (int i=0; i<strs.size(); i++)
    {
        int p = atoi(strs[i].c_str());
        CHECK_GT(length_, p);
        data[p] = 1;
    }
    boost::split(strs, line.second, boost::is_any_of(","));
    CHECK_EQ(strs.size(), 2);
    Dtype noclick = atoi(strs[0].c_str());
    Dtype click = atoi(strs[1].c_str());
    label = click / (click + noclick);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  TxtDataParameter txt_data_param = this->layer_param_.txt_data_param();
  const int batch_size = txt_data_param.batch_size();
  length_ = txt_data_param.length();
  const int new_height = 1;
  const int new_width = 1;

  vector<int> top_shape(4, 1);
  top_shape[1] = length_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  memset(prefetch_data, 0, sizeof(Dtype)*batch->data_.count());
  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    ParseLines(lines_[lines_id_], this->transformed_data_, prefetch_label[item_id])
    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.txt_data_param().shuffle()) {
        ShuffleLines();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TxtDataLayer);
REGISTER_LAYER_CLASS(TxtData);

}  // namespace caffe
#endif  // USE_OPENCV
