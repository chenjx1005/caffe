import caffe
import numpy as np
from scipy.sparse import csr_matrix

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr = csr_matrix((data, indices, indptr), shape=(3, 10)).toarray()

net = caffe.Net("sparse_net.prototxt", caffe.TRAIN)
net.blobs['value'].data[...] = data
net.blobs['indices'].data[...] = indices
net.blobs['ptr'].data[...] = indptr

#check forward pass
weight = net.params['spfc'][0].data
y = csr.dot(weight)
y_net = net.forward()['spfc']
error = y - y_net
assert (np.abs(error) < 0.001).all()

#check backward pass
top_diff = net.blobs['spfc'].diff
top_diff[...] = np.random.rand(*top_diff.shape)
diff = csr.T.dot(top_diff)
net.backward()
diff_net = net.params['spfc'][0].diff
error = diff - diff_net
assert (np.abs(error) < 0.001).all()