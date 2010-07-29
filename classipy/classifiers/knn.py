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

    def train(self, label_values, converted=False):
        """Stores the training data internally.

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
        labels, values = zip(*list(label_values))
        self._labels = labels
        self._values = values
        return self

    def predict(self, value, converted=False):
        """Evaluates a single value against the training data.

        Args:
            value: List-like object with same dimensionality used for training
                or the result of using convert_value if converted=True.
            converted: If True then the input is in the correct internal format.

        Returns:
            Sorted (descending) list of (confidence, label).
        """
        if not converted:
            value = self.convert_value(value)
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
        inv_k = 1. / self._k
        return [(x * inv_k, y) for x, y in out]


def main():
    print(__doc__)

if __name__ == '__main__':
    main()
