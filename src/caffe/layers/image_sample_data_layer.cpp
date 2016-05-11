#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_sample_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageSampleDataLayer<Dtype>::~ImageSampleDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageSampleDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  class_num_ = this->layer_param_.image_sample_data_param().class_num();
  const int new_height = this->layer_param_.image_sample_data_param().new_height();
  const int new_width  = this->layer_param_.image_sample_data_param().new_width();
  const bool is_color  = this->layer_param_.image_sample_data_param().is_color();
  string root_folder = this->layer_param_.image_sample_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the files of each class with filenames and labels
  lines_.resize(class_num_);
  lines_id_.resize(class_num_);
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream insource(source.c_str());
  string filename;
  vector<string> files;
  while(insource >> filename) {
    files.push_back(filename);
  }
  CHECK_EQ(files.size(), class_num_) << "class_num and number of files must be equal.";

  int label;
  for (int i=0; i<class_num_; i++) {
    const string& file = files[i];
    LOG(INFO) << "Opening file " << file;
    std::ifstream infile(file.c_str());
    while (infile >> filename >> label) {
      lines_[i].push_back(std::make_pair(filename, label));
    }
  }
  
  // randomly shuffle data
  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleClass();

  unsigned int amount = 0;
  for (int i=0; i<class_num_; i++) {
    amount += lines_[i].size();
    lines_size_.push_back(lines_[i].size());
  }
  LOG(INFO) << "A total of " << amount << " images.";

  class_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_sample_data_param().rand_skip()) {
    for (int i=0; i<class_num_; i++) {
      unsigned int skip = caffe_rng_rand() %
          this->layer_param_.image_sample_data_param().rand_skip();
      CHECK_GT(lines_[i].size(), skip) << "Not enough points to skip";
      lines_id_[i] = skip;
    }
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[0][lines_id_[0]].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[0][lines_id_[0]].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_sample_data_param().batch_size();
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
void ImageSampleDataLayer<Dtype>::ShuffleClass() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  shuffle(lines_id_.begin(), lines_id_.end(), prefetch_rng);
  shuffle(lines_size_.begin(), lines_size_.end(), prefetch_rng);
}

template <typename Dtype>
void ImageSampleDataLayer<Dtype>::ShuffleImages(int i) {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_[i].begin(), lines_[i].end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageSampleDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageSampleDataParameter image_data_param = this->layer_param_.image_sample_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[class_id_][lines_id_[class_id_]].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[class_id_][lines_id_[class_id_]].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size_[class_id_], lines_id_[class_id_]);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[class_id_][lines_id_[class_id_]].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[class_id_][lines_id_[class_id_]].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[class_id_][lines_id_[class_id_]].second;
    // go to the next iter
    lines_id_[class_id_]++;
    if (lines_id_[class_id_] >= lines_size_[class_id_]) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching of class " << class_id_ << " from start.";
      lines_id_[class_id_] = 0;
      ShuffleImages(class_id_);
    }
    class_id_++;
    if (class_id_ >= class_num_) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting classes prefetching from start.";
      class_id_ = 0;
      ShuffleClass();
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSampleDataLayer);
REGISTER_LAYER_CLASS(ImageSampleData);

}  // namespace caffe
#endif  // USE_OPENCV
