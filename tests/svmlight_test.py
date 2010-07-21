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

import classipy


class Test(unittest.TestCase):
    def test_traintest0(self):
        # Data
        train_labels = [-1, -1, 1, 1, 1]
        train_values = [[0.], [1.], [2.], [3.], [4.]]
        #train_values = map(np.array, train_values)
        # Test
        c = classipy.SVMLight()
        train_label_values = zip(train_labels, train_values)
        c.train(train_label_values)
        self.assertEqual(c.predict([2.])[0][1], 1)
        self.assertEqual(c.predict([4.])[0][1], 1)
        self.assertEqual(c.predict([0.])[0][1], -1)

    def testa_traintest1(self):
        # Data
        train_labels = [-1] * 50 + [1] * 50
        train_values = np.random.normal(loc=-5, size=50).tolist()
        train_values += np.random.normal(loc=5, size=50).tolist()
        train_values = [[x] for x in train_values]
        #print(train_values)
        # Test
        c = classipy.SVMLinear({'B': '1'})
        train_label_values = zip(train_labels, train_values)
        c.train(train_label_values)
        self.assertEqual(c.predict([-5.])[0][1], -1)
        self.assertEqual(c.predict([-10.])[0][1], -1)
        self.assertEqual(c.predict([5.])[0][1], 1)
        self.assertEqual(c.predict([10.])[0][1], 1)

if __name__ == '__main__':
    unittest.main()
