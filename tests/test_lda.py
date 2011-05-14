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
from IPython.Shell import IPShellEmbed

class Test(unittest.TestCase):
    def test_linsep2d(self):
        def sample0():
            return np.random.multivariate_normal([-5., -5.], [[1, 0], [0, 1.]])
        def sample1():
            return np.random.multivariate_normal([5., 5.], [[1, 0], [0, 1.]])
        # Data
        sam = 10
        train_labels = [-1] * sam + [1] * sam
        train_values = [sample0() for x in range(sam)]
        train_values += [sample1() for x in range(sam)]
        c = classipy.LDA()
        train_label_values = zip(train_labels, train_values)
        c.train(train_label_values)
        def display():
            import matplotlib.pyplot as mp
            mp.scatter(*zip(*train_values[:sam]), c='r')
            mp.scatter(*zip(*train_values[sam:]), c='b')
            mp.figure()
            mp.scatter(c._values[:sam], [0] * sam, c='r')
            mp.scatter(c._values[sam:], [0] * sam, c='b')
            mp.show()
        #display()
        self.assertEqual(c.predict(sample0())[0][1], -1)
        self.assertEqual(c.predict(sample1())[0][1], 1)

    def test_linsep1d(self):
        def sample0():
            return np.random.multivariate_normal([-5.], [[1.]])
        def sample1():
            return np.random.multivariate_normal([5.], [[1.]])
        # Data
        sam = 10
        train_labels = [-1] * sam + [1] * sam
        train_values = [sample0() for x in range(sam)]
        train_values += [sample1() for x in range(sam)]
        c = classipy.LDA()
        train_label_values = zip(train_labels, train_values)
        c.train(train_label_values)
        def display():
            import matplotlib.pyplot as mp
            mp.scatter(train_values[:sam], [0] * sam, c='r')
            mp.scatter(train_values[sam:], [0] * sam, c='b')
            mp.figure()
            mp.scatter(c._values[:sam], [0] * sam, c='r')
            mp.scatter(c._values[sam:], [0] * sam, c='b')
            mp.show()
        self.assertEqual(c.predict(sample0())[0][1], -1)
        self.assertEqual(c.predict(sample1())[0][1], 1)

    def test_linsep10d(self):
        def sample0():
            return np.random.multivariate_normal([-5.] * 10, np.eye(10))
        def sample1():
            return np.random.multivariate_normal([5.] * 10, np.eye(10))
        # Data
        sam = 10
        train_labels = [-1] * sam + [1] * sam
        train_values = [sample0() for x in range(sam)]
        train_values += [sample1() for x in range(sam)]
        c = classipy.LDA()
        train_label_values = zip(train_labels, train_values)
        c.train(train_label_values)
        def display():
            import matplotlib.pyplot as mp
            mp.scatter(c._values[:sam], [0] * sam, c='r')
            mp.scatter(c._values[sam:], [0] * sam, c='b')
            mp.show()
        #display()
        self.assertEqual(c.predict(sample0())[0][1], -1)
        self.assertEqual(c.predict(sample1())[0][1], 1)

    def test_notlinsep2d(self):
        def sample0():
            return np.random.multivariate_normal([33., 30.], [[1, -1], [0, 1.]])
        def sample1():
            return np.random.multivariate_normal([35., 35.], [[1, 1], [0, 1.]])
        # Data
        sam = 100
        train_labels = [-1] * sam + [1] * sam
        train_values = [sample0() for x in range(sam)]
        train_values += [sample1() for x in range(sam)]
        c = classipy.LDA()
        train_label_values = zip(train_labels, train_values)
        c.train(train_label_values)
        def display():
            import matplotlib.pyplot as mp
            mp.scatter(*zip(*train_values[:sam]), c='r')
            mp.scatter(*zip(*train_values[sam:]), c='b')
            xsc = mp.xlim()
            ysc = mp.ylim()
            line = [(c._mean_pca + (np.array([-1, 1]) * c._proj * x)[0]).tolist() for x in [-1000, 1000]]
            mp.plot(*zip(*line), c='g')
            mp.xlim(xsc)
            mp.ylim(ysc)
            mp.figure()
            val = np.dot(train_values - c._mean_pca, c._proj.T)
            mp.scatter(val[:sam], [0] * sam, c='r')
            mp.scatter(val[sam:], [0] * sam, c='b')
            mp.show()
        #display()
        self.assertEqual(c.predict(sample0())[0][1], -1)
        self.assertEqual(c.predict(sample1())[0][1], 1)




if __name__ == '__main__':
    unittest.main()
