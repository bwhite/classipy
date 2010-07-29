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
    def linsep(self, classifier_name, classifier, sam, dim):
        sam_mean = [100.] * dim
        def sample0():
            return np.abs(np.random.multivariate_normal(sam_mean, np.eye(dim)))
        def sample1():
            return -sample0()
        test_values0 = [sample0() for x in range(sam)]
        test_values1 = [sample1() for x in range(sam)]
        train_labels = [-1] * sam + [1] * sam
        train_values = [sample0() for x in range(sam)]
        train_values += [sample1() for x in range(sam)]
        train_label_values = zip(train_labels, train_values)
        st = time.time()
        c = classifier().train(train_label_values)
        train_time = time.time() - st
        st = time.time()
        for x in test_values0:
            self.assertEqual(c.predict(x)[0][1], -1)
        for x in test_values1:
            self.assertEqual(c.predict(x)[0][1], 1)
        predict_time = time.time() - st
        print('Lin - Train:[%d] Test:[%d] Dim:[%d] - Train:[%f] Pred:[%f] - [%s]' % (sam * 2, sam * 2, dim, train_time, predict_time, classifier_name))

    def test_linsep2d(self):
        def sample2():
            return np.random.multivariate_normal([1., 1.], [[1, 0], [0, 1.]])
        def sample3():
            return -sample2()
        # Data
        def classifiers():
            for x in dir(classipy):
                classifier = getattr(classipy, x)
                try:
                    if issubclass(classifier, classipy.BinaryClassifier):
                        if classifier != classipy.BinaryClassifier:
                            yield x, classifier
                except TypeError:
                    pass
        for sam, dim in [(2, 2), (10, 2), (100, 10), (100, 100)]:
            for classifier_name, classifier in classifiers():
                self.linsep(classifier_name, classifier, sam, dim)

if __name__ == '__main__':
    unittest.main()
