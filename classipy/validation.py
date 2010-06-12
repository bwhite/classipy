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

"""Validation tools (e.g., cross validation, metrics)
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np


def cross_validation(classifier, labels, values, folds=10, options=None):
    """Performs cross validation on a BinaryClassifier.

    Args:
        classifier: A classifier that conforms to the BinaryClassifier spec.
        labels: List of integer labels.
        values: List of list-like objects, all with the same dimensionality.
        folds: Number of partitions to split the data into (default 10).
    Returns:
        A dictionary of performance statistics.
    """
    # Randomly shuffle the data
    labels_values = zip(labels, values)
    random.shuffle(labels_values)
    # Split up folds
    fold_size = int(np.ceil(len(labels) / float(folds)))
    folds = [labels_values[x * fold_size:(x + 1) * fold_size]
             for x in range(folds)]
    # Iterate, leaving one fold out for testing each time
    accuracy_sum = 0.
    for test_num in range(folds):
        train_labels_values = sum(folds[:test_num] + folds[test_num + 1:], [])
        c = classifier(options=options)
        c.train(*zip(*train_labels_values))
        test_results = [(label, c.test(value)[0][1])
                        for label, value in folds[test_num]]
        accuracy = len([1 for x in test_results if x[0] == x[1]])
        accuracy /= float(len(test_results))
        print(accuracy)
