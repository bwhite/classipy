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

import itertools
import random

import numpy as np


def cross_validation(classifier_class, labels, values, num_folds=10, options=None):
    """Performs cross validation on a BinaryClassifier.

    Args:
        classifier_class: A classifier that conforms to the BinaryClassifier spec.
        labels: List of integer labels.
        values: List of list-like objects, all with the same dimensionality.
        num_folds: Number of partitions to split the data into (default 10).
    Returns:
        A dictionary of performance statistics.
    """
    # Randomly shuffle the data
    labels_values = zip(labels, values)
    random.shuffle(labels_values)
    # Split up folds
    fold_size = int(np.ceil(len(labels) / float(num_folds)))
    folds = [labels_values[x * fold_size:(x + 1) * fold_size]
             for x in range(num_folds)]
    # Iterate, leaving one fold out for testing each time
    accuracy_sum = 0.
    for test_num in range(num_folds):
        train_labels_values = sum(folds[:test_num] + folds[test_num + 1:], [])
        c = classifier_class(options=options)
        c.train(*zip(*train_labels_values))
        out = evaluate(c, *zip(*folds[test_num]))
        print(out)
        accuracy_sum += out['accuracy']
    return accuracy_sum / num_folds


def _confusion_stats(confusion):
    """Generates statistics given a square confusion matrix.

    Args:
        confusion: A square sparse confusion matrix in the form of a dict of
            dicts such that confusion[true_label][pred_label].  All values are
            expected to be integers, missing values are taken as zeros.
    Returns:
        A dictionary of performance statistics (precision, recall, accuracy)
    """
    overall_total = 0.
    overall_correct = 0.
    precision = {}
    recall = {}
    for true_label in confusion:
        # Generate base level statistics
        row_sum = sum(confusion[true_label].values())
        col_sum = sum([confusion[x][true_label]
                       for x in confusion if confusion[x].has_key(true_label)])
        tp = confusion[true_label][true_label] # Num True == Predict == cur class
        fn = row_sum - tp # Num True == cur class and Predict != cur class
        fp = col_sum - tp # Num True != cur class and Predict == cur class
        overall_correct += tp
        overall_total += row_sum
        # Generate relevant output statistics
        try:
            precision[true_label] = tp / float(tp + fp)
        except ZeroDivisionError:
            precision[true_label] = float('nan')
        try:
            recall[true_label] = tp / float(tp + fn)
        except ZeroDivisionError:
            recall[true_label] = float('nan')
    accuracy = overall_correct / float(overall_total)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}


def evaluate(classifier, labels, values, class_selector=None):
    """Classifies the provided values and generates stats based on  the labels.

    Args:
        classifier: A classifier instance that conforms to the BinaryClassifier
            spec.
        labels: List of integer labels.
        values: List of list-like objects, all with the same dimensionality.
        class_selector: Function that takes classifier output and returns a
            label. If None (default) then use first class label (highest
            confidence).
    Returns:
        A dictionary of performance statistics.
    """
    if class_selector == None:
        class_selector = lambda x: x[0][1]
    test_results = [(label, class_selector(classifier.predict(value)))
                    for label, value in zip(labels, values)]
    confusion = {}
    # Generate confusion matrix [true_label][pred_label]
    for true_label, pred_label in test_results:
        try:
            confusion[true_label][pred_label] += 1
        except KeyError:
            try:
                confusion[true_label][pred_label] = 1
            except KeyError:
                confusion[true_label] = {pred_label: 1}
    return _confusion_stats(confusion)


def multi_evaluate(classifiers, labels, values, class_selectors=None):
    """Classifies the provided values and generates stats based on  the labels.

    Args:
        classifiers: A list of classifiers that conforms to the BinaryClassifier spec.
        labels: List of integer labels.
        values: List of list-like objects, all with the same dimensionality.
        class_selectors: List of functions (one per classifier, if less then
            reuse last) that take classifier output and return label. If None
            (default) then use first class label (highest confidence).

    Returns:
        A dictionary of performance statistics.
    """
    if class_selectors == None:
        class_selectors = [lambda x: x[0][1]]
    classifier_output = []
    for sel, cls in itertools.izip_longest(class_selectors, classifiers):
        classifier_output.append(evaluate(cls, labels, values, sel))
    """TODO
    1. For binary classifiers generate full comparison lists for ROC/PR curves
    2. Generate confusion matrices using highest confidence metric
    3. 
    """
    return {'classifier_output': classifier_output}
