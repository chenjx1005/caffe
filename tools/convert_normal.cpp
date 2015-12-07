#include <algorithm>
#include <fstream>  
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <boost/thread/mutex.hpp>

#include "boost/scoped_ptr.hpp"
#include <boost/algorithm/string.hpp>
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  
using std::pair;
using boost::scoped_ptr;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  std::ifstream infile(argv[1]);
  const char* db_path = argv[2];
  const int LENGTH = atoi(argv[3]);
  const int record_num = atoi(argv[4]);
  
  scoped_ptr<db::DB> db(db::GetDB("lmdb"));
  db->Open(db_path, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  
  int count = 0;
  const int kMaxKeyLength = 256;
  boost::mutex write_mutex;
  boost::mutex read_mutex;
  //bool mt_finish_flag = false;
  long long line_id = 0;
#pragma omp parallel for shared(line_id, read_mutex, write_mutex, db, txn, count)
  for (int p = 0; p < record_num; p++) {
    SparseDatum datum;
    char key_cstr[kMaxKeyLength];
    std::string cur_line;
    long long tmp_line_id;
    read_mutex.lock();
    if( !infile.eof() ) {
        if(!std::getline(infile, cur_line)) {
            //mt_finish_flag = true;
            continue;
        }
        tmp_line_id = line_id++;
    }
    else {
        read_mutex.unlock();
        continue;
    }
    read_mutex.unlock(); 
    int length = snprintf(key_cstr, kMaxKeyLength, "%020lld", tmp_line_id);
    // to do list, fill datum
    std::string out;
    std::istringstream iss(cur_line);
    string groupkey;
    string noclick_click;
    string data;
    iss >> groupkey >> noclick_click >> data;
    vector<string> strs;
    boost::split(strs, noclick_click, boost::is_any_of(","));
    float click = atoi(strs[1].c_str());
    float noclick = atoi(strs[0].c_str());
    datum.set_label(click / (click + noclick));
    datum.set_size(LENGTH);
    boost::split(strs, data, boost::is_any_of(","));
    for (int i=0; i<strs.size(); i++) {
      datum.add_indices(atoi(strs[i].c_str())-1);
      datum.add_data(1);
    }
    datum.set_nnz(strs.size());
    CHECK(datum.SerializeToString(&out));

    write_mutex.lock();
    txn->Put(string(key_cstr, length), out);
    if (++count % 1000 == 0) {
        
        LOG(ERROR) << "Processed " << count << " files.";
        txn->Commit();
        txn.reset(db->NewTransaction());
    }
    write_mutex.unlock();
  }
  
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
