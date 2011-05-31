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
        # Train
        values = list(np.random.random((100, 1)))
        values += list(.5 + np.random.random((100, 1)))
        labels = [0] * 100
        labels += [1] * 100
        label_values = zip(labels, values)
        # Test - K=1
        print(classipy.cross_validation(classipy.KNN, label_values))


if __name__ == '__main__':
    unittest.main()
