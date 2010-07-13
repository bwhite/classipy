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

"""K-Nearest Neighbor Classifier
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np

from base import BinaryClassifier

class KNN(BinaryClassifier):
    def __init__(self, options=None):
        super(KNN, self).__init__()
        try:
            self._k = options['k']
        except (KeyError, TypeError):
            self._k = 1
        self.to_type = np.ndarray

    def train(self, labels, values):
        """Stores the training data internally.

        Args:
            labels: List of integer labels.
            values: List of list-like objects, all with the same dimensionality.
        Returns:
            self
        """
        values = self._convert_values(values)
        self._labels = labels
        self._values = values
        return self

    def predict(self, value):
        """Evaluates a single value against the training data.

        Args:
            value: List-like object with same dimensionality used for training.

        Returns:
            Sorted (descending) list of (confidence, label).
        """
        value = self._convert_value(value)
        dists = value - self._values
        dists = np.sum(dists * dists, 1)
        dists_labels = zip(dists, self._labels)
        dists_labels.sort()
        out = {}
        for dist, label in dists_labels[:self._k]:
            try:
                out[label] += 1
            except KeyError:
                out[label] = 1
        out = [x[::-1] for x in out.items()]
        out.sort(reverse=True)
        return out


def main():
    print(__doc__)

if __name__ == '__main__':
    main()
