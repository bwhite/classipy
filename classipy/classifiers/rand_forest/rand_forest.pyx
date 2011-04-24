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
http://cvlab.epfl.ch/~lepetit/papers/lepetit_cvpr05.pdf
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np
cimport numpy as np
import operator
import cPickle as pickle

cdef extern from "fast_hist.h":
    void fast_histogram(int *labels, int labels_size, int *hist)
    double fast_entropy(double *hist, int hist_size)

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

    cdef np.ndarray[np.int32_t, ndim=1] histogram(self, np.ndarray[np.int32_t, ndim=1] labels):
        """Computes a histogram of labels

        Args:
            labels:  Ndarray of labels (ints) (must be 0 <= x < num_classes)

        Returns:
            Ndarray of length 'num_classes' with indexes as labels and
            values as counts
        """
        # We are touching pointers here, so check the input before this
        # if one of the labels goes out of bounds there will be a problem.
        cdef np.ndarray out = np.zeros(self.num_classes, dtype=np.int32)
        if not labels.size:
            return out
        cdef int *out_p = <int *>out.data
        cdef int *labels_p = <int *>labels.data
        cdef int labels_dim = labels.shape[0]
        fast_histogram(labels_p, labels_dim, out_p)
        return out

    cdef np.ndarray[np.float64_t, ndim=1] normalized_histogram(self, np.ndarray[np.int32_t, ndim=1] labels):
        """Computes a histogram of labels

        Args:
            labels:  Ndarray of labels (ints) (must be 0 <= x < num_classes)

        Returns:
            Ndarray of length 'num_classes' with indexes as labels and
            values as probs
        """
        cdef np.ndarray out = self.histogram(labels)
        cdef double scale = 1./ np.sum(out)
        return scale * out

    cdef double entropy(self, np.ndarray[np.float64_t, ndim=1] q):
        """Shannon Entropy

        Args:
            q: Ndarray of length 'num_classes' with indexes as labels and
                values as probabilities

        Returns:
            Entropy in 'bits'
        """
        return fast_entropy(<double*>q.data, <int>q.shape[0])

    cdef double information_gain(self, np.ndarray[np.int32_t, ndim=1] ql,
                                        np.ndarray[np.int32_t, ndim=1] qr):
        """Compute the information gain of the split

        Args:
            ql: Ndarray of length 'num_classes' with indexes as labels and
                values as counts
            qr: Ndarray of length 'num_classes' with indexes as labels and
                values as counts
        """
        cdef double sum_l = np.sum(ql)  # NOTE: Possible overflow opportunity
        cdef double sum_r = np.sum(qr)
        if not sum_l or not sum_r:
            return 0.
        h_q = self.entropy((ql + qr) / (sum_l + sum_r))
        h_ql = self.entropy(ql / sum_l)
        h_qr = self.entropy(qr / sum_r)
        pr_l = sum_l / (sum_l + sum_r)
        pr_r = 1. - pr_l
        return h_q - pr_l * h_ql - pr_r * h_qr

    cdef make_features(self, seed):
        """Generate a list of length self.num_feat features

        Args:
            seed: An integer seed given to np.random.seed, if 0 then don't seed

        Returns:
            List of (feat_num, feat_ser, feat) in order of increasing feat_num
        """
        if not seed:
            np.random.seed(seed)
        out = []
        for feat_num in range(self.num_feat):
            feat_ser = self.gen_feature()
            feat = self.make_feature_func(feat_ser)
            out.append((feat_ser, feat))
        return out

    cpdef train_map_hists(self, labels, values, seed=0):
        """Compute the histograms for a series of labels/values

        Args:
            labels: Np array of labels
            values: List of opaque values (given to the feature)
            seed: An integer seed given to np.random.seed, if 0 then don't seed
                (default is 0)

        Returns:
            Tuple of (qls, qrs, func_sers)
            where qls/qrs is a num_feat X num_class array and
            func_sers is length num_feat.
        """
        qls, qrs, func_sers = [], [], []
        for func_ser, func in self.make_features(seed):
            ql, qr = self.partition_labels(labels, values, func)
            qls.append(self.histogram(ql))
            qrs.append(self.histogram(qr))
            func_sers.append(func_ser)
        return np.vstack(qls), np.vstack(qrs), func_sers

    cpdef train_combine_hists(self, qls_qrs_func_sers_iter):
        """Sum the intermediate results among splits

        Args:
            qls_qrs_func_sers_iter: Iterator of num_splits elements of
                (qls, qrs, func_sers) where
                qls/qrs is num_feat X num_class array
                func_sers is length num_feat

        Returns:
            Tuple of (qls, qrs, func_sers) with qls/qrs of shape
                num_feat x num_class_array and func_sers length of num_feat
        """
        qlss, qrss, func_serss = zip(*list(qls_qrs_func_sers_iter))
        # Verify that all funcs are the same
        for func_sers in func_serss[1:]:
            assert func_serss[0] == func_sers
        cdef np.ndarray total_qls = np.sum(qlss, 0)  # num_feat x num_class_array
        cdef np.ndarray total_qrs = np.sum(qrss, 0)  # num_feat x num_class_array
        return total_qls, total_qrs, func_serss[0]

    cpdef train_reduce_info(self, qls, qrs, func_sers):
        """
        Args:
            qls: Left histogram of shape num_feat x num_class_array
            qrs: Right histogram of shape num_feat x num_class_array
            func_sers: List of serialized functions

        Returns:
            Tuple of (info_gain, func_ser)
        """
        info_gains = [self.information_gain(ql, qr)
                      for ql, qr in zip(qls, qrs)]
        max_ind = np.argmax(info_gains)
        return info_gains[max_ind], func_sers[max_ind]

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
        info_gain, func_ser = self.train_reduce_info(*self.train_map_hists(labels, values))
        if info_gain <= self.min_info:
            return (self.normalized_histogram(labels),)
        if self.feature_to_str:
            print('MaxInfo: Feat[%s] InfoGain[%f]' % (self.feature_to_str(func_ser),
                                                      info_gain))
        tree_depth = tree_depth - 1
        func = self.make_feature_func(func_ser)
        ql_labels, ql_values, qr_labels, qr_values = self.partition_label_values(labels, values, func)
        return (func_ser,
                self.train_find_feature(ql_labels, ql_values, tree_depth),
                self.train_find_feature(qr_labels, qr_values, tree_depth),
                {'info_gain': info_gain})

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
