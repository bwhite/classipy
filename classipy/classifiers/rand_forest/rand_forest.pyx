#!/usr/bin/env python
# (C) Copyright 2010 Brandyn A. White
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Random Forest Classifier
See: http://research.microsoft.com/pubs/145347/BodyPartRecognition.pdf
and: http://cvlab.epfl.ch/~lepetit/papers/lepetit_cvpr05.pdf
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np
cimport numpy as np
import multiprocessing
import Queue
import operator
import cPickle as pickle

cdef extern from "fast_hist.h":
    void fast_histogram(int *labels, int labels_size, int *hist)
    double fast_entropy(double *hist, int hist_size)
    void depth_predict(np.uint16_t *depth, double *out_prob, np.uint8_t *out_ind, np.int32_t *trees, np.int32_t *links, double *leaves,
                   double *u, double *v, np.int32_t *t, int num_trees, int num_nodes, int num_leaves, int num_classes)


cdef class RandomForestClassifier(object):
    cdef public object make_feature_func
    cdef public object gen_feature
    cdef int num_feat
    cdef int tree_depth
    cdef double min_info
    cdef int num_classes
    cdef int num_trees
    cdef public object trees_ser
    cdef public object trees
    cdef public object feature_to_str
    cdef int num_procs
    
    def __init__(self, make_feature_func, gen_feature, feature_to_str=None, num_classes=2, num_feat=100, tree_depth=4,
                 min_info=.01, num_trees=1, num_procs=1, trees_ser=None):
        # From args (these must be provided again during deserialization)
        self.make_feature_func = make_feature_func  # Takes a string feat to func
        self.gen_feature = gen_feature  # Makes string representation of feature
        self.feature_to_str = feature_to_str  # If available, use for debugging
        # From args (everything here must be in __reduce__ for pickle to work)
        self.num_classes = num_classes
        self.num_feat = num_feat
        self.tree_depth = tree_depth
        self.min_info = min_info
        self.num_trees = num_trees
        self.num_procs = num_procs
        self.trees_ser = trees_ser if trees_ser else []
        # Derived from args
        self.trees = []

    def __reduce__(self):
        return RandomForestClassifier, (None, None, None,
                                        self.num_classes, self.num_feat,
                                        self.tree_depth, self.min_info,
                                        self.num_trees, self.num_procs,
                                        self.trees_ser)

    def dumps(self):
        """Serializes the classifier to a string

        Returns:
            A string that can be passed to the class' loads method
        """
        return pickle.dumps(self, -1)

    @classmethod
    def loads(cls, s, make_feature_func, gen_feature=None, feature_to_str=None):
        """Returns a classifier instance given a serialized form

        Args:
            s: Serialized string

        Returns:
            An instance of this class as it was before it was serialized
        """
        out = pickle.loads(s)
        out.make_feature_func = make_feature_func
        out.gen_feature = gen_feature
        out.feature_to_str = feature_to_str
        out.trees = [out.tree_deserialize(x)
                     for x in out.trees_ser]
        return out

    cdef partition_labels(self, labels, values, func):
        """
        Args:
            labels: Iterator of ints
            values: Iterator of vecs
            func: func(vec) = Boolean

        Returns:
            Tuple of (ql, qr)
            ql: Elements of vecs s.t. func is false
            qr: Elements of vecs s.t. func is true
        """
        ql_ind, qr_ind = [], []
        for ind, value in enumerate(values):
            if func(value):
                qr_ind.append(ind)
            else:
                ql_ind.append(ind)
        return labels[ql_ind], labels[qr_ind]

    cdef partition_label_values(self, labels, values, func):
        """
        Args:
            labels: Iterator of ints
            values: Iterator of vecs
            func: func(vec) = Boolean

        Returns:
            Tuple of (ql, qr)
            ql: Elements of vecs s.t. func is false
            qr: Elements of vecs s.t. func is true
        """
        ql_ind, qr_ind = [], []
        ql_val, qr_val = [], []
        for ind, value in enumerate(values):
            if func(value):
                qr_ind.append(ind)
                qr_val.append(value)
            else:
                ql_ind.append(ind)
                ql_val.append(value)
        return labels[ql_ind], ql_val, labels[qr_ind], qr_val

    cdef np.ndarray[np.float64_t, ndim=1] normalized_histogram(self,np.ndarray[np.int32_t, ndim=1] labels):
        """Computes a normalized histogram of labels

        Args:
            labels:  Ndarray of labels (ints) (must be 0 <= x < num_classes)

        Returns:
            Ndarray with indexes as labels and values as prob.
        """
        if not labels.size:  # Nothing provided
            return
        # We are touching pointers here, so check the input before this
        # if one of the labels goes out of bounds there will be a problem.
        cdef np.ndarray out = np.zeros(self.num_classes, dtype=np.int32)
        cdef int *out_p = <int *>out.data
        cdef int *labels_p = <int *>labels.data
        cdef int labels_dim = labels.shape[0]
        fast_histogram(labels_p, labels_dim, out_p)
        norm = 1. / len(labels)
        return norm * out

    cdef double entropy(self, np.ndarray[np.int32_t, ndim=1] q):
        """Shannon Entropy

        Args:
            q: Np array of integral labels

        Returns:
            Entropy in 'bits'
        """
        cdef np.ndarray hist = self.normalized_histogram(q)
        if hist == None:
            return 0.
        return fast_entropy(<double*>hist.data, <int>hist.shape[0])

    cdef double information_gain(self, np.ndarray[np.int32_t, ndim=1] ql,
                                 np.ndarray[np.int32_t, ndim=1] qr):
        """Compute the information gain of the split

        Args:
            ql: Np array of labels
            qr: Np array of labels
        """
        h_q = self.entropy(np.concatenate((ql, qr)))
        h_ql = self.entropy(ql)
        h_qr = self.entropy(qr)
        pr_l = len(ql) / float(len(ql) + len(qr))
        pr_r = 1. - pr_l
        return h_q - pr_l * h_ql - pr_r * h_qr

    cpdef train_find_max_feature(self, labels, values, num_feat, queue=None):
        """

        Args:
            labels: Np array of labels
            values: List of opaque values (given to the feature)
            num_feat: Number of features to compute
            queue: If present then the return value should be put here (default: None)

        Returns:
            info_gain, serialized function
        """
        cdef float max_info_gain = -float('inf')
        max_info_gain_func = None
        max_info_gain_func_ser = None
        for feat_num in range(num_feat):
            func_ser = self.gen_feature()
            func = self.make_feature_func(func_ser)
            ql, qr = self.partition_labels(labels, values, func)
            info_gain = self.information_gain(ql,
                                              qr)
            if self.feature_to_str:
                print('Feat[%s] InfoGain[%f]' % (self.feature_to_str(func_ser),
                                                 info_gain))
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_func_ser = func_ser
        out = max_info_gain, max_info_gain_func_ser
        if queue:
            queue.put(out)
        return out


    cdef train_find_feature(self, labels, values, tree_depth):
        """Recursively and greedily train the tree

        Args:
            labels: Np array of labels
            values: List of opaque values (given to the feature)
            tree_depth: Num of levels more in tree (<0 terminates)

        Returns:
            (prob array, ) if leaf else
            (func_ser, left_tree(false), right_tree(true), metadata)
        """
        if tree_depth < 0:
            return (self.normalized_histogram(labels),)
        cdef float max_info_gain = -float('inf')
        if self.num_procs <= 1 or len(labels) < 10:
            max_info_gain, max_info_gain_func_ser = self.train_find_max_feature(labels, values, self.num_feat)
        else:
            queue = multiprocessing.Queue()
            ps = [multiprocessing.Process(target=self.train_find_max_feature,
                                          args=(labels, values, self.num_feat / self.num_procs, queue))
                  for x in range(self.num_procs)]
            for p in ps: p.start()
            max_info_gain, max_info_gain_func_ser = max([queue.get() for p in ps], key=operator.itemgetter(0))
            for p in ps: p.join()
        if max_info_gain <= self.min_info:
            return (self.normalized_histogram(labels),)
        if self.feature_to_str:
            print('MaxInfo: Feat[%s] InfoGain[%f]' % (self.feature_to_str(max_info_gain_func_ser),
                                                      max_info_gain))
        tree_depth = tree_depth - 1
        max_info_gain_func = self.make_feature_func(max_info_gain_func_ser)
        ql_labels, ql_values, qr_labels, qr_values = self.partition_label_values(labels, values, max_info_gain_func)
        return (max_info_gain_func_ser,
                self.train_find_feature(ql_labels, ql_values, tree_depth),
                self.train_find_feature(qr_labels, qr_values, tree_depth),
                {'info_gain': max_info_gain})

    def train(self, label_values):
        """Train the classifier

        If num_trees > 1, the data is split evenly and contiguously.  This
        lets you control which data each tree gets.  Data not evenly divisible
        is unused.

        Args:
            label_values: List of (label, values) where
                0 <= label < num_classes.  The value can be anything as it is
                only given to the provided functions.

        Return:
            self

        Raises:
            ValueError: When make_feature_func or gen_feature are None.
        """
        if not self.make_feature_func or not self.gen_feature:
            raise ValueError('make_feature_func and gen_feature must be set!')
        labels, values = zip(*label_values)
        labels = np.array(labels, dtype=np.int32)
        self.num_classes = max(self.num_classes, np.max(labels) + 1)  # Requires we get one sample per class
        assert np.min(labels) >= 0
        assert np.max(labels) < self.num_classes
        self.trees_ser, self.trees = [], []
        s = len(labels) / self.num_trees  # Samples per tree (round down, leftovers not trained on)
        for t in range(self.num_trees):
            self.trees_ser.append(self.train_find_feature(labels[t * s: (t + 1) * s],
                                                          values[t * s: (t + 1) * s],
                                                          self.tree_depth))
            self.trees.append(self.tree_deserialize(self.trees_ser[-1]))
        # TODO: We may want to learn the probabilities on the entire dataset
        return self

    cdef predict_tree(self, value, tree):
        """Perform prediction using a tree recursively

        Args:
            value: The value to be classified (given to the features)
            tree: Tree of the form (recursive)
                (func, left_tree(false), right_tree(true), metadata)
                until the leaf nodes which are (prob array, )
        Returns:
            Prob array belonging to the leaf we end up in
        """
        assert len(tree) == 4 or len(tree) == 1
        if len(tree) != 4:
            return tree[0]
        if tree[0](value):
            return self.predict_tree(value, tree[2])
        return self.predict_tree(value, tree[1])

    def predict(self, value):
        """Perform classifier prediction using model

        Args:
            value: The value to be classified (given to the features)

        Returns:
            List of (prob, label) descending by probability
        """
        mean_pred = np.mean([self.predict_tree(value, t) for t in self.trees], 0)
        out = [x[::-1] for x in enumerate(mean_pred)]
        out.sort(reverse=True)
        return out

    cpdef tree_deserialize(self, tree_ser):
        """Given a tree_ser, gives back a tree

        Args:
            tree_ser: Tree of the form (recursive)
                (func_ser, left_tree(false), right_tree(true), metadata)
                until the leaf nodes which are (prob array, )

        Returns:
            Same structure except func_ser is converted to func using
            make_feature_func.

        Raises:
            ValueError: When make_feature_func or gen_feature are None.
        """
        if not self.make_feature_func:
            raise ValueError('make_feature_func must be set!')
        if len(tree_ser) != 4:
            return tree_ser
        return (self.make_feature_func(tree_ser[0]),
                self.tree_deserialize(tree_ser[1]),
                self.tree_deserialize(tree_ser[2]),
                tree_ser[3])


# NOTE: These are just to simplify building, they will be moved later
cdef class FastClassifier(object):
    # Below are used for updating the trees
    cdef int node_counter
    cdef int leaf_counter
    cdef object temp_u
    cdef object temp_v
    cdef object temp_t
    cdef object temp_leaves
    cdef object temp_links
    # Below are used for storing the trees
    cdef np.ndarray trees # [trees]
    cdef np.ndarray u  # [nodes, 2] y/x vals
    cdef np.ndarray v  # [nodes, 2]  y/x vals
    cdef np.ndarray t  # [nodes] for all nodes
    cdef np.ndarray leaves  # [leaves, num_classes]
    cdef np.ndarray links  # [nodes, 2] false/true paths
    cdef int num_trees
    cdef int num_nodes
    cdef int num_leaves
    cdef int num_classes

    def __init__(self, trees_ser):
        self.update_trees(trees_ser)

    cdef make_feature_func(self, feat_str):
        data = pickle.loads(feat_str)
        u = data['u']
        v = data['v']
        t = data['t']
        return u, v, t

    cpdef update_trees(self, trees_ser):
        self.node_counter = 0
        self.leaf_counter = 0
        self.temp_u, self.temp_v, self.temp_t = [], [], []
        self.temp_leaves = []
        self.temp_links = []
        self.trees = np.array([self.tree_deserialize(x)
                               for x in trees_ser], dtype=np.int32)
        self.leaves = np.array(self.temp_leaves, dtype=np.float64).ravel()
        self.links = np.array(self.temp_links, dtype=np.int32).ravel()
        self.u = np.array(self.temp_u, dtype=np.float64).ravel()
        self.v = np.array(self.temp_v, dtype=np.float64).ravel()
        self.t = np.array(self.temp_t, dtype=np.int32)
        self.num_trees = len(self.trees)
        self.num_nodes = len(self.temp_u)
        self.num_leaves = len(self.temp_leaves)
        self.num_classes = len(self.temp_leaves[0])

    cpdef tree_deserialize(self, tree_ser):
        """Given a tree_ser, gives back a tree

        Args:
            tree_ser: Tree of the form (recursive)
                (func_ser, left_tree(false), right_tree(true), metadata)
                until the leaf nodes which are (prob array, )
            make_feature_func: 

        Returns:
            Same structure except func_ser is converted to func using
            make_feature_func.
        """
        if len(tree_ser) != 4:
            val = self.leaf_counter
            self.leaf_counter += 1
            self.temp_leaves.append(tree_ser[0])
            print('Leaf[%d] = %s' % (val, self.temp_leaves[-1]))
            return -val - 1
        val = self.node_counter
        self.node_counter += 1
        u, v, t = self.make_feature_func(tree_ser[0])
        self.temp_u.append(u)
        self.temp_v.append(v)
        self.temp_t.append(t)
        self.temp_links.append([self.tree_deserialize(tree_ser[1]),
                                self.tree_deserialize(tree_ser[2])])
        return val

    def predict(self, np.ndarray[np.uint16_t, ndim=2] depth_image):
        cdef np.ndarray depth = depth_image.ravel()
        cdef np.ndarray out_ind = np.zeros(depth_image.size, dtype=np.uint8)
        cdef np.ndarray out_prob = np.zeros(depth_image.size, dtype=np.float64)
        print('Predicting')
        depth_predict(<np.uint16_t *>depth.data, <double *>out_prob.data, <np.uint8_t *>out_ind.data, <np.int32_t *>self.trees.data, <np.int32_t *>self.links.data, <double *>self.leaves.data, <double *>self.u.data, <double *>self.v.data, <np.int32_t *>self.t.data, self.num_trees, self.num_nodes, self.num_leaves, self.num_classes)
        return out_ind.reshape((480, 640)), out_prob.reshape((480, 640))
