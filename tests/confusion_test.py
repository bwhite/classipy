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

import classipy.validation


class Test(unittest.TestCase):
    def atest_confusion0(self):
        test_in = {0: {0: 1, 1: 0},
                   1: {0: 1, 1: 0}}
        out  = classipy.validation._confusion_stats(test_in)
        print(out)
        self.assertEqual(out['precision'][0], .5)
        # Should not equal self as it is a NaN
        self.assertTrue(np.isnan(out['precision'][1]))
        self.assertEqual(out['recall'], {0: 1., 1: 0.})
        test_in = {0: {0: 10, 1: 5},
                   1: {0: 0, 1: 5}}
        out  = classipy.validation._confusion_stats(test_in)
        self.assertEqual(out['precision'], {0: 1., 1: 0.5})
        self.assertEqual(out['recall'], {0: 10/15., 1: 1.})

    def test_confusion1(self):
        # Example from
        # http://www.compumine.com/web/public/newsletter/20071/precision-recall
        test_in = {0: {0: 25, 1: 5, 2: 2},
                   1: {0: 3, 1: 32, 2: 4},
                   2: {0: 1, 1: 0, 2: 15}}
        out  = classipy.validation._confusion_stats(test_in)
        print(out)
        self.assertEqual(out['precision'][0], 25. / (25 + 3 + 1))
        self.assertEqual(out['recall'][0],  25. / (25 + 5 + 2))

if __name__ == '__main__':
    unittest.main()
