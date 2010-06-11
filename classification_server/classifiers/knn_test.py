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

import numpy as np

import knn


class Test(unittest.TestCase):
    def test_traintest0(self):
        # Data
        train_labels = [0, 1, 1, 1, 0]
        train_values = [[0.], [1.], [2.], [3.], [4.], [5.]]
        train_values = map(np.array, train_values)
        test_value = np.array([2.])
        # Test - K=1
        test_output = [(1, 1)]
        c = knn.KNN()
        c.train(train_labels, train_values)
        self.assertEqual(c.predict(test_value), test_output)
        # Test - K=3
        test_output = [(3, 1)]
        c = knn.KNN(options={'k': 3})
        c.train(train_labels, train_values)
        self.assertEqual(c.predict(test_value), test_output)
        # Test - K=5
        test_output = [(3, 1), (2, 0)]
        c = knn.KNN(options={'k': 5})
        c.train(train_labels, train_values)
        self.assertEqual(c.predict(test_value), test_output)



if __name__ == '__main__':
    unittest.main()
