// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers. Especially suit for a LISTFILE with many repeats.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::map;
using std::set;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::map<string, string> maps;
  std::string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;

  //save unopened files name
  std::set<string> unopened;
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    // check if this image has been processed
    std::map<string, string>::iterator it;
    it = maps.find(lines[line_id].first);
    if (it != maps.end()) {
      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
          lines[line_id].first.c_str());

      txn->Put(string(key_cstr, length), it->second);
    }
    else {
      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
        // Guess the encoding type from the file name
        string fn = lines[line_id].first;
        size_t p = fn.rfind('.');
        if ( p == fn.npos )
          LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
        enc = fn.substr(p);
        std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
      }
      status = ReadImageToDatum(root_folder + lines[line_id].first,
          lines[line_id].second, resize_height, resize_width, is_color,
          enc, &datum);
      if (status == false) {
        unopened.insert(lines[line_id].first);
        continue;
      }
      if (check_size) {
        if (!data_size_initialized) {
          data_size = datum.channels() * datum.height() * datum.width();
          data_size_initialized = true;
        } else {
          const std::string& data = datum.data();
          CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
              << data.size();
        }
      }
      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
          lines[line_id].first.c_str());

      // Put in db
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(string(key_cstr, length), out);
      // insert this image into maps
      maps.insert(std::make_pair(lines[line_id].first, out));
      if (maps.size() % 1000 == 0) {
        LOG(INFO) << maps.size() << "unique files have been processed.";
      }
    }

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  // LOG the unopened files
  LOG(INFO) <<"these files can't be opened.";
  set<string>::iterator it;
  for (it = unopened.begin(); it != unopened.end(); it++) {
      LOG(INFO) << *it;
  }
  // LOG the imageset size
  LOG(INFO) << maps.size() <<"Unique files.";
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
