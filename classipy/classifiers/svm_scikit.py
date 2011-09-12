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

import math
import tempfile
import cPickle as pickle
import numpy as np

from base import BinaryClassifier


class SVMScikit(BinaryClassifier):

    def __init__(self, **kw):
        super(SVMScikit, self).__init__()
        self._param = kw
        self._labels = None

    def _make_model(self):
        from scikits.learn import svm
        kw = dict(self._param)
        kw.setdefault('kernel', 'linear')  # NOTE(brandyn): Default to linear
        self._m = svm.SVC(**kw)

    def train(self, label_values, converted=False):
        """Build a model.

        Args:
        label_values: Iterable of tuples of label and list-like objects
            Example: [(label, value), ...]
            or the result of using convert_label_values if converted=True.
        converted: If True then the input is in the correct internal format.
        Returns:
            self
        """
        if not converted:
            label_values = self.convert_label_values(label_values)
        try:
            labels, values = zip(*list(label_values))
        except ValueError:
            raise ValueError('label_values is empty')
        self._labels = [np.min(labels), np.max(labels)]
        self._make_model()
        self._m.fit(np.array(values), np.array(labels))
        return self

    def predict(self, value, converted=False):
        """Evaluates a single value against the training data.

        NOTE: Confidence is currently set to 0!

        Args:
            value: List-like object with same dimensionality used for training
                or the result of using convert_value if converted=True.
            converted: True then the input is in the correct internal format.

        Returns:
            Sorted (descending) list of (confidence, label)
        """
        if not converted:
            value = self.convert_value(value)
        val = self._m.decision_function(np.array([value]))[0]
        if val >= 0:
            return [(float(np.abs(val)), self._labels[1])]
        return [(float(np.abs(val)), self._labels[0])]


def main():
    print(__doc__)

if __name__ == '__main__':
    main()
