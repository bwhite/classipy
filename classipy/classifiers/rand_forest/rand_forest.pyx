# cython: profile=True
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
import random

cdef extern from "fast_hist.h":
    void fast_histogram(int *labels, int labels_size, int *hist)
    void fast_histogram_weight(int *labels, int *weights, int labels_size, int weight_rows, int num_classes, int *hist)
    double fast_entropy(double *hist, int hist_size)

# Include base features, see these if you want to implement your own
include "vector_feature.pyx"


cpdef np.ndarray[np.int32_t, ndim=1] histogram(np.ndarray[np.int32_t, ndim=1, mode='c'] labels, int num_classes):
    """Computes a histogram of labels

    Args:
        labels:  Ndarray of labels (ints) (must be 0 <= x < num_classes)

    Returns:
        Ndarray of length 'num_classes' with indexes as labels and
        values as counts
    """
    # We are touching pointers here, so check the input before this
    # if one of the labels goes out of bounds there will be a problem.
    cdef np.ndarray out = np.zeros(num_classes, dtype=np.int32)
    if not labels.size:
        return out
    cdef int *out_p = <int *>out.data
    cdef int *labels_p = <int *>labels.data
    cdef int labels_dim = labels.shape[0]
    fast_histogram(labels_p, labels_dim, out_p)
    return out


cpdef np.ndarray[np.int32_t, ndim=2, mode='c'] histogram_weight(np.ndarray[np.int32_t, ndim=1, mode='c'] labels, np.ndarray[np.int32_t, ndim=2, mode='c'] weights, int num_classes):
    """Computes a histogram of labels

    Args:
        labels:  Ndarray of labels (ints) (must be 0 <= x < num_classes)
        weights: Ndarray of weights (ints) that signify how much to weight each label
            each row results in a different histogram.

    Returns:
        Ndarray of shape (weight_rows x num_classes) with indexes as labels and
        values as counts
    """
    # We are touching pointers here, so check the input before this
    # if one of the labels goes out of bounds there will be a problem.
    cdef np.ndarray out = np.zeros(num_classes * weights.shape[0], dtype=np.int32).reshape((weights.shape[0], num_classes))
    out = np.ascontiguousarray(out)
    if not labels.size:
        return out
    fast_histogram_weight(<int *>labels.data,  <int *>weights.data, labels.shape[0], weights.shape[0], num_classes, <int *>out.data)
    return out


cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] normalized_histogram(np.ndarray[np.int32_t, ndim=1, mode='c'] labels, int num_classes):
    """Computes a histogram of labels

    Args:
        labels:  Ndarray of labels (ints) (must be 0 <= x < num_classes)

    Returns:
        Ndarray of length 'num_classes' with indexes as labels and
        values as probs
    """
    cdef np.ndarray out = histogram(labels, num_classes)
    cdef double scale = 1./ np.sum(out)
    return scale * out


cdef class RandomForestClassifier(object):
    cdef public object feature_factory
    cdef int tree_depth
    cdef double min_info
    cdef int num_classes
    cdef int num_feat
    cdef public object trees_ser
    cdef public object trees
    
    def __init__(self, feature_factory, num_classes=2, tree_depth=4, num_feat=1000,
                 min_info=.01, trees_ser=None):
        self.feature_factory = feature_factory
        # From args (everything here must be in __reduce__ for pickle to work)
        self.num_classes = num_classes
        self.num_feat = num_feat
        self.tree_depth = tree_depth
        self.min_info = min_info
        self.trees_ser = trees_ser if trees_ser else []
        # Derived from args
        self.trees = []

    cdef make_features(self, seed=0):
        """Generate a list of features

        Args:
            seed: An integer seed given to np.random.seed, if 0 then don't seed
                (default is 0)

        Returns:
            List of (feat_ser, feat) in order of increasing feat_num
        """
        if seed:
            np.random.seed(seed)
        return [self.feature_factory.gen_feature()
                for x in range(self.num_feat)]

    cpdef train_map_hists(self, labels, values, feats):
        """Compute the histograms for a series of labels/values

        Args:
            labels: Np array of labels
            values: List of opaque values (given to the feature)
            feats: List of functions

        Returns:
            Tuple of (qls, qrs)
            where qls/qrs is a num_feat X num_class array
        """
        qls, qrs = [], []
        for feat in feats:
            cur_qls, cur_qrs = feat.label_histograms(labels, values, self.num_classes)
            qls.append(cur_qls)
            qrs.append(cur_qrs)
        return np.vstack(qls), np.vstack(qrs)

    cpdef train_combine_hists(self, qls_qrs_iter):
        """Sum the intermediate results among splits

        Args:
            qls_qrs_iter: Iterator of num_splits elements of
                (qls, qrs) where
                qls/qrs is num_feat X num_class array

        Returns:
            Tuple of (qls, qrs) with qls/qrs of shape
                num_feat x num_class_array 
        """
        qlss, qrss = zip(*list(qls_qrs_iter))
        cdef np.ndarray total_qls = np.sum(qlss, 0)  # num_feat x num_class_array
        cdef np.ndarray total_qrs = np.sum(qrss, 0)  # num_feat x num_class_array
        return total_qls, total_qrs

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] entropy(self, np.ndarray[np.float64_t, ndim=2, mode='c'] q):
        """Shannon Entropy

        Args:
            q: Ndarray of shape 'num_features X num_classes' with indexes as labels and
                values as probabilities

        Returns:
            np array of entropy in 'bits'
        """
        cdef int i
        cdef np.ndarray x
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] out = np.zeros(q.shape[0])
        for i, x in enumerate(q):
            out[i] = fast_entropy(<double*>x.data, <int>x.shape[0])
        return out

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] information_gain(self, np.ndarray[np.int32_t, ndim=2, mode='c'] qls,
                                                                     np.ndarray[np.int32_t, ndim=2, mode='c'] qrs):
        """Compute the information gain of the split

        Args:
            qls: Left histogram of shape num_feat x num_class_array
            qrs: Right histogram of shape num_feat x num_class_array
        """
        cdef np.ndarray sum_l = np.sum(qls, 1)  # NOTE: Possible overflow opportunity
        cdef np.ndarray sum_r = np.sum(qrs, 1)
        out = np.zeros(sum_l.shape[0])
        inds = ((sum_l != 0) & (sum_r != 0)).nonzero()[0]
        qls = qls[inds]
        qrs = qrs[inds]
        sum_l = np.asfarray(sum_l[inds]).reshape(len(inds), 1)
        sum_r = np.asfarray(sum_r[inds]).reshape(len(inds), 1)
        cdef np.ndarray qrls_norm = (qls + qrs) / (sum_l + sum_r)
        cdef np.ndarray h_q = self.entropy(qrls_norm)
        cdef np.ndarray h_ql = self.entropy(qls / sum_l)
        cdef np.ndarray h_qr = self.entropy(qrs / sum_r)
        pr_l = (sum_l / (sum_l + sum_r)).flatten()
        pr_r = 1. - pr_l
        out[inds] = h_q - pr_l * h_ql - pr_r * h_qr
        return out

    cpdef train_reduce_info(self, np.ndarray[np.int32_t, ndim=2, mode='c'] qls, np.ndarray[np.int32_t, ndim=2, mode='c'] qrs):
        """
        Args:
            qls: Left histogram of shape num_feat x num_class_array
            qrs: Right histogram of shape num_feat x num_class_array

        Returns:
            Tuple of (info_gain, feat_ind)
        """
        cdef np.ndarray info_gains = self.information_gain(qls, qrs)
        max_ind = np.argmax(info_gains)
        return info_gains[max_ind], max_ind

    cdef train_find_feature(self, labels, values, tree_depth):
        """Recursively and greedily train the tree

        Args:
            labels: Np array of labels
            values: List of opaque values (given to the feature)
            tree_depth: Num of levels more in tree (<0 terminates)

        Returns:
            (prob array, ) if leaf else
            (feat_ser, left_tree(false), right_tree(true), metadata)
        """
        if tree_depth < 0:
            return (normalized_histogram(labels, self.num_classes),)
        feats = self.make_features()
        info_gain, feat_ind = self.train_reduce_info(*self.train_map_hists(labels, values, feats))
        feat = self.feature_factory.select_feature(feats, feat_ind)        
        feat_ser = feat.dumps()
        print('[%d]MaxInfo: Feat[%s] InfoGain[%f]' % (tree_depth, feat,
                                                      info_gain))
        if info_gain <= self.min_info:
            return (normalized_histogram(labels, self.num_classes),)
        tree_depth = tree_depth - 1
        ql_labels, ql_values, qr_labels, qr_values = feat.label_values_partition(labels, values)
        return (feat_ser,
                self.train_find_feature(ql_labels, ql_values, tree_depth),
                self.train_find_feature(qr_labels, qr_values, tree_depth),
                {'info_gain': info_gain})

    def train(self, label_values, replace=True, converted=False):
        """Train the classifier

        Args:
            label_values: List of (label, values) where
                0 <= label < num_classes.  The value can be anything as it is
                only given to the provided functions.

        Return:
            self

        Raises:
            ValueError: When feature_factory is not set.
        """
        if not self.feature_factory:
            raise ValueError('feature_factory must be set!')
        labels, values = zip(*label_values)
        labels = np.array(labels, dtype=np.int32)
        try:
            values = np.array(values)
        except ValueError:
            pass
        self.num_classes = max(self.num_classes, np.max(labels) + 1)  # Requires we get one sample per class
        assert np.min(labels) >= 0
        assert np.max(labels) < self.num_classes
        if replace:
            self.trees_ser, self.trees = [], []    
        self.trees_ser.append(self.train_find_feature(labels,
                                                      values,
                                                      self.tree_depth))
        self.trees.append(self.tree_deserialize(self.trees_ser[-1]))
        return self

    cdef predict_tree(self, value, tree):
        """Perform prediction using a tree recursively

        Args:
            value: The value to be classified (given to the features)
            tree: Tree of the form (recursive)
                (feat, left_tree(false), right_tree(true), metadata)
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

    def predict(self, value, converted=False):
        """Perform classifier prediction using model

        Args:
            value: The value to be classified (given to the features)

        Returns:
            List of (prob, label) descending by probability

        Raises:
            ValueError: Classifier is not trained
        """
        if not self.trees:
            raise ValueError('Classifier must be trained to predict')
        mean_pred = np.mean([self.predict_tree(value, t) for t in self.trees], 0)
        out = [x[::-1] for x in enumerate(mean_pred)]
        out.sort(reverse=True)
        return out

    cpdef tree_deserialize(self, tree_ser):
        """Given a tree_ser, gives back a tree

        Args:
            tree_ser: Tree of the form (recursive)
                (feat_ser, left_tree(false), right_tree(true), metadata)
                until the leaf nodes which are (prob array, )

        Returns:
            Same structure except feat_ser is converted to feat using
            make_feature_func.

        Raises:
            ValueError: When make_feature_func or gen_feature are None.
        """
        if not self.feature_factory:
            raise ValueError('feature_factory must be set!')
        if len(tree_ser) != 4:
            return tree_ser
        return (self.feature_factory.loads(tree_ser[0]),
                self.tree_deserialize(tree_ser[1]),
                self.tree_deserialize(tree_ser[2]),
                tree_ser[3])

    def _hist_to_str(self, hist):
        hist = list(enumerate(hist))
        hist.sort(key=lambda x: x[1], reverse=True)
        return ' '.join(['%s:%.4g' % x for x in hist])


    def graphviz(self):
        gvs = []
        for tree in self.trees:
            graphviz_ctr = [0]
            node_names, links = self._graphviz_tree_recurse(tree, graphviz_ctr)
            gvs.append('digraph{%s}' % ';'.join(node_names + links))
        return gvs

    def graphviz_google(self):
        return ['https://chart.googleapis.com/chart?cht=gv:dot&chl=%s' % x for x in self.graphviz()]

    def _graphviz_tree_recurse(self, tree, graphviz_ctr, parent='', left_node=False):
        cur_id = str(graphviz_ctr[0])
        graphviz_ctr[0] += 1
        color = 'red' if left_node else 'green'
        if len(tree) == 1:  # Leaf
            cur_name = '%s[label="%s"]' % (cur_id, self._hist_to_str(tree[0]))
            node_names, links = [cur_name], []
            links.append('%s->%s[color=%s]' % (parent, cur_id, color))
            return node_names, links
        cur_name = '%s[label="I[%f]P[%s]"]' % (cur_id, tree[3]['info_gain'],
                                               str(tree[0]))
        node_names = [cur_name]
        links = []
        if parent:
            links.append('%s->%s[color=%s]' % (parent, cur_id, color))
        child_node_names, child_links = self._graphviz_tree_recurse(tree[1],
                                                                    graphviz_ctr,
                                                                    parent=cur_id,
                                                                    left_node=True)
        node_names.extend(child_node_names)
        links.extend(child_links)
        child_node_names, child_links = self._graphviz_tree_recurse(tree[2],
                                                                    graphviz_ctr,
                                                                    parent=cur_id,
                                                                    left_node=False)
        node_names.extend(child_node_names)
        links.extend(child_links)
        return node_names, links
