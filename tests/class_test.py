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
"""Test
"""
__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import unittest
import time

import numpy as np

import classipy

class Test(unittest.TestCase):
    def evaluate_classifiers(self, classifier_name, classifier, neg_sam_train, pos_sam_train, neg_sam_test, pos_sam_test, test_name, sample0, sample1, no_fail=False):
        test_values0 = [sample0() for x in range(neg_sam_test)]
        test_values1 = [sample1() for x in range(pos_sam_test)]
        dim = len(test_values0[0])
        train_labels = [-1] * neg_sam_train + [1] * pos_sam_train
        train_values = [sample0() for x in range(neg_sam_train)]
        train_values += [sample1() for x in range(pos_sam_train)]
        train_label_values = zip(train_labels, train_values)
        st = time.time()
        c = classifier().train(train_label_values)
        train_time = time.time() - st
        st = time.time()
        def test_pred(test_values, expected):
            errors = 0
            for x in test_values:
                out = c.predict(x)
                self.assertTrue(isinstance(out, list))
                self.assertTrue(isinstance(out[0], tuple))
                self.assertTrue(isinstance(out[0][0], float))
                self.assertTrue(isinstance(out[0][1], int))
                try:
                    self.assertEqual(out[0][1], expected)
                except AssertionError, e:
                    if no_fail:
                        errors += 1
                    else:
                        raise e
            return errors
        errors = test_pred(test_values0, -1)
        errors += test_pred(test_values1, 1)
        predict_time = time.time() - st
        print('%s - Train:[%d, %d] Test:[%d, %d] Dim:[%d] - Train:[%f] Pred:[%f] Accuracy:[%f]- [%s]' % (test_name, neg_sam_train, pos_sam_train,
                                                                                                         neg_sam_test, pos_sam_test, dim, train_time,
                                                                                                         predict_time,
                                                                                                         1 - errors / float(pos_sam_test + neg_sam_test),
                                                                                                         classifier_name))

    def test_linsep2d(self):
        def classifiers():
            for x in dir(classipy):
                classifier = getattr(classipy, x)
                try:
                    if issubclass(classifier, classipy.BinaryClassifier):
                        if classifier != classipy.BinaryClassifier:
                            yield x, classifier
                except TypeError:
                    pass
        for sam, dim in [(2, 2), (10, 2), (100, 10), (100, 100), (2, 100)]:
            sam_mean = [100.] * dim
            def sample0():
                return np.abs(np.random.multivariate_normal(sam_mean, np.eye(dim)))
            def sample1():
                return -sample0()
            for classifier_name, classifier in classifiers():
                self.evaluate_classifiers(classifier_name, classifier, sam, sam, sam, sam, 'lin', sample0, sample1)

    def test_perfect(self):
        def classifiers():
            for x in dir(classipy):
                classifier = getattr(classipy, x)
                try:
                    if issubclass(classifier, classipy.BinaryClassifier):
                        if classifier != classipy.BinaryClassifier:
                            yield x, classifier
                except TypeError:
                    pass
        def sample0():
            # Perfect dimension plus a tiny bit of noise
            return np.array([-1.]) + np.random.random() / 10000.
        def sample1():
            return -sample0()
        for classifier_name, classifier in classifiers():
            self.evaluate_classifiers(classifier_name, classifier, 100, 100, 100, 100, 'per', sample0, sample1)

    def test_perfect_with_random(self):
        def classifiers():
            for x in dir(classipy):
                classifier = getattr(classipy, x)
                try:
                    if issubclass(classifier, classipy.BinaryClassifier):
                        if classifier != classipy.BinaryClassifier:
                            yield x, classifier
                except TypeError:
                    pass
        for dim in [0, 1, 10, 100, 1000]:
            def sample0():
                out = np.random.random(dim + 1)
                out[0] = -1. + (np.random.random() - .5) / 10000
                return out
            def sample1():
                out = np.random.random(dim + 1)
                out[0] = 1. + (np.random.random() - .5) / 10000
                return out
            for classifier_name, classifier in classifiers():
                self.evaluate_classifiers(classifier_name, classifier, 1000, 1, 1000, 1000, 'per+rnd', sample0, sample1, no_fail=True)



if __name__ == '__main__':
    unittest.main()
