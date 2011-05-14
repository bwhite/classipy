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
import zlib

EXCEPTIONS = ['SVMLight', 'SVMLinear']

class Test(unittest.TestCase):
    def test_serialize(self):
        def classifiers():
            for x in dir(classipy):
                classifier = getattr(classipy, x)
                try:
                    if issubclass(classifier, classipy.BinaryClassifier):
                        if classifier != classipy.BinaryClassifier:
                            yield x, classifier
                except TypeError:
                    pass
        dim = 10
        n_tr = 50
        n_te = 1000
        sam_mean = [100] * dim
        def sample0():
            return np.abs(np.random.multivariate_normal(sam_mean, np.eye(dim)))
        def sample1():
            return -sample0()
        for classifier_name, classifier in classifiers():
            if classifier_name in EXCEPTIONS:
                print('Skipping [%s]' % classifier_name)
                continue
            print(classifier_name)
            # We use the same sampler for both classes (better test)
            test_label_values = [(-1, sample0()) for x in range(n_te)]
            test_label_values += [(1, sample0()) for x in range(n_te)]
            train_label_values = [(-1, sample0()) for x in range(n_tr)]
            train_label_values += [(1, sample0()) for x in range(n_tr)]
            train_label_values2 = [(-1, sample1()) for x in range(n_tr)]
            train_label_values2 += [(1, sample1()) for x in range(n_tr)]
            c = classifier().train(train_label_values)
            c_same = classifier().train(train_label_values)
            c2 = classifier().train(train_label_values2)
            self.assertNotEqual(c.dumps(), c2.dumps())
            self.assertEquals(c.dumps(), c.dumps())
            # Removed as this is not globally true (breaks LibLinear for some reason)
            #self.assertEquals(c.dumps(), c_same.dumps())
            self.assertEquals(classifier.loads(c.dumps()).dumps(), c.dumps())
            self.assertTrue(isinstance(c.dumps(), str))
            print('Length Uncomp[%d] ZlibComp[%d]' % (len(c.dumps()), len(zlib.compress(c.dumps()))))
            c_dumped = classifier.loads(c.dumps())
            for label, value in test_label_values:
                self.assertEquals(c.predict(value), c_dumped.predict(value))
            

if __name__ == '__main__':
    unittest.main()
