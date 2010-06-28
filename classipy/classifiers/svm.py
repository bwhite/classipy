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

"""SVM Classifier
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np

import libsvm.svm
import libsvm.svmutil

from base import BinaryClassifier


class SVM(BinaryClassifier):
    def __init__(self, options=None):
        super(SVM, self).__init__()
        try:
            self._param = ' '.join(['-%s %s' % x for x in options.items()])
        except AttributeError:
            self._param = ''
        self._param += ' -q'  # Makes silent
        self.to_type = list

    def _train_normalize(self, values):
        """Learns the min/max for a series of values, setting _shift, _scale.

        Args:
            values: List of list-like objects, all with the same dimensionality.
        """
        values = self._convert_values(values)
        values = np.array(values)
        mi = np.min(values, 0)
        ma = np.max(values, 0)
        self._shift = mi
        self._scale = 1 / (ma - mi)
        self._scale[np.isinf(self._scale)] = 0.  # Force inf to zero
        return [self._normalize(x) for x in values]

    def _normalize(self, value):
        """Normalizes a value [0, 1]

        Args:
            values: List-like object.
        Returns:
            Normalized value
        """
        value = np.array(value)
        value -= self._shift
        value *= self._scale
        return self._convert_value(value)
        
    def train(self, labels, values):
        """Build a model.

        Args:
            labels: List of integer labels
            values: List of list-like objects, all with the same dimensionality.
        """
        values = self._train_normalize(values)
        prob  = libsvm.svm.svm_problem(labels, values)
        param = libsvm.svm.svm_parameter(self._param)
        self._m = libsvm.svmutil.svm_train(prob, param)

    def predict(self, value):
        """Evaluates a single value against the training data.

        NOTE: Confidence is currently set to 0!

        Args:
            value: List-like object with same dimensionality used for training.

        Returns:
            Sorted (descending) list of (confidence, label)
        """
        value = self._normalize(value)
        labels, stats, confidence = libsvm.svmutil.svm_predict([-1], [value], self._m)
        print(labels)
        return [(0., labels[0])]

def main():
    print(__doc__)

if __name__ == '__main__':
    main()
