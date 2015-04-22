#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/integer_blob.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class LookupTableLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LookupTableLayerTest()
      : blob_bottom_(new IntegerBlob<Dtype>(2, 1, 3, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    int* text_data  = blob_bottom_->mutable_cpu_indices();
    for (int i = 0; i < 6; i++) {
      text_data[i] = i * 10;
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LookupTableLayerTest() { delete blob_bottom_; delete blob_top_; }
  IntegerBlob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LookupTableLayerTest, TestDtypesAndDevices);

TYPED_TEST(LookupTableLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LookupTableParameter* lookup_table_param =
      layer_param.mutable_lookup_table_param();
  lookup_table_param->set_size(5);
  lookup_table_param->set_n_index(100);
  shared_ptr<LookupTableLayer<Dtype> > layer(
      new LookupTableLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
}

TYPED_TEST(LookupTableLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LookupTableParameter* lookup_table_param =
        layer_param.mutable_lookup_table_param();
    lookup_table_param->set_size(5);
    lookup_table_param->set_n_index(100);
    lookup_table_param->mutable_weight_filler()->set_type("uniform");
    lookup_table_param->mutable_weight_filler()->set_min(1);
    lookup_table_param->mutable_weight_filler()->set_max(2);
    shared_ptr<LookupTableLayer<Dtype> > layer(
        new LookupTableLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
      EXPECT_LE(data[i], 2.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(LookupTableLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LookupTableParameter* lookup_table_param =
        layer_param.mutable_lookup_table_param();
    lookup_table_param->set_size(5);
    lookup_table_param->set_n_index(100);
    lookup_table_param->mutable_weight_filler()->set_type("gaussian");
    LookupTableLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, -2);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
