#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/integer_blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class IndexDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  IndexDataLayerTest()
      : backend_(DataParameter_DB_LEVELDB),
        blob_top_data_(new IntegerBlob<Dtype>()),
        blob_top_data2_(new IntegerBlob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        seed_(1701) {}
  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_data2_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  // Fill the LevelDB with data: if unique_pixels, each pixel is unique but
  // all images are the same; else each image is unique but all pixels within
  // an image are the same.
  void Fill(DataParameter_DB backend) {
    backend_ = backend;
    LOG(INFO) << "Using temporary dataset " << *filename_;
    scoped_ptr < db::DB > db(db::GetDB(backend));
    db->Open(*filename_, db::NEW);
    scoped_ptr < db::Transaction > txn(db->NewTransaction());
    for (int i = 0; i < 5; ++i) {
      IndexDatum datum;
      datum.set_label(i);
      datum.set_n_index(100);
      datum.set_height_data(2);
      datum.set_size(10);

      for (int j = 0; j < 10; ++j) {
              datum.mutable_data()->Add(2 * j * i);
              datum.mutable_data()->Add(2 * j * i + 1);
              datum.mutable_indices()->Add(j * (i +1));
      }
      stringstream ss;
      ss << i;
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(ss.str(), out);
    }
    txn->Commit();
    db->Close();
  }

  void TestRead() {
    LayerParameter param;
    IndexDataParameter* data_param = param.mutable_index_data_param();
    data_param->set_batch_size(5);
    data_param->set_backend(backend_);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    IndexDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 2);
    EXPECT_EQ(blob_top_data_->height(), 10);
    EXPECT_EQ(blob_top_data_->width(), 1);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 10; ++j) {
           EXPECT_EQ((i+1) * j, blob_top_data_->cpu_indices()[j + i * 10])
               << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
      for (int i = 0; i < 5; ++i) {
              for (int j = 0; j < 10; ++j) {
                 EXPECT_EQ(2 * j * i,
                           blob_top_data_->cpu_data()[2 * j + i * 20])
                     << "debug: iter " << iter << " i " << i << " j " << j;
                 EXPECT_EQ(2 * j * i + 1,
                           blob_top_data_->cpu_data()[2* j + 1 + i * 20])
                    << "debug: iter " << iter << " i " << i << " j " << j;
              }
            }
    }
  }

  virtual ~IndexDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  DataParameter_DB backend_;
  shared_ptr<string> filename_;
  IntegerBlob<Dtype>* const blob_top_data_;
  IntegerBlob<Dtype>* const blob_top_data2_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(IndexDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(IndexDataLayerTest, TestReadLevelDB) {
  this->Fill(DataParameter_DB_LEVELDB);
  std::cout << "filled level db\n";
  this->TestRead();
}

TYPED_TEST(IndexDataLayerTest, TestReadLMDB) {
  this->Fill(DataParameter_DB_LMDB);
  this->TestRead();
}


}  // namespace caffe
