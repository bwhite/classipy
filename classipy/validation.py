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


def cross_validation(classifier_class, label_values, num_folds=10,
                     options=None, expand_options=False, converted=False):
    """Performs cross validation on a BinaryClassifier.

    The same partitions will be produced if random.seed is used before this is
    called.  Loads all label_values in memory to group them.

    Args:
        classifier_class: A classifier that conforms to BinaryClassifier spec
        label_values: Iterable of tuples of label and list-like objects.
            Example: [(label, value), ...]
        num_folds: Number of partitions to split the data into (default 10).
            If len(label_values) < num_folds, then we use len(label_values)
        options: Options to pass to the classifier
        expand_options: True then use classifier_class(**options) else use
            classifier_class(options=options)
        converted: True then the input is in the correct internal format.

    Returns:
        Accuracy
    """
    # Randomly shuffle the data
    label_values = list(label_values)
    random.shuffle(label_values)
    if len(label_values) < num_folds:
        num_folds = len(label_values)
    # Split up folds
    fold_size = int(np.ceil(len(label_values) / float(num_folds)))
    folds = [label_values[x * fold_size:(x + 1) * fold_size]
             for x in range(num_folds)]
    # Iterate, leaving one fold out for testing each time
    accuracy_sum = 0.
    for test_num in range(num_folds):
        train_labels_values = sum(folds[:test_num] + folds[test_num + 1:], [])
        if expand_options:
            c = classifier_class(**options)
        else:
            c = classifier_class(options=options)
        c.train(train_labels_values, converted=converted)
        out = evaluate(c, folds[test_num], converted=converted)
        accuracy_sum += out['accuracy']
    return accuracy_sum / num_folds


def confusion_stats(confusion):
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
    f1 = {}
    miss_rate = {}
    tps = {}
    fns = {}
    fps = {}
    total_true = {}
    total_pred = {}
    overall_total = sum([sum(x.values()) for x in confusion.values()])
    for true_label in confusion:
        # Generate base level statistics
        # row_sum is num of true examples for the cur label
        row_sum = sum(confusion[true_label].values())
        # col_sum is num of predicted examples for the cur label
        col_sum = sum([confusion[x][true_label]
                       for x in confusion if true_label in confusion[x]])
        try:
            # Num True == Predict == cur class
            tp = confusion[true_label][true_label]
        except KeyError:
            tp = 0
        fn = row_sum - tp  # Num True == cur class and Predict != cur class
        fp = col_sum - tp  # Num True != cur class and Predict == cur class
        total_true[true_label] = row_sum
        total_pred[true_label] = col_sum
        tps[true_label] = tp
        fps[true_label] = fp
        fns[true_label] = fn
        overall_correct += tp
        # Generate relevant output statistics
        try:
            precision[true_label] = tp / float(tp + fp)
        except ZeroDivisionError:
            precision[true_label] = float('nan')
        try:
            recall[true_label] = tp / float(tp + fn)
        except ZeroDivisionError:
            recall[true_label] = float('nan')
        f1[true_label] = 2. * precision[true_label] * recall[true_label]
        try:
            f1[true_label] /= (precision[true_label] + recall[true_label])
        except ZeroDivisionError:
            f1[true_label] = float('nan')
        miss_rate[true_label] = 1 - recall[true_label]
    try:
        accuracy = overall_correct / float(overall_total)
    except ZeroDivisionError:
        accuracy = float('nan')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'tp': tps, 'fp': fps, 'fn': fns, 'total_true': total_true,
            'total_pred': total_pred, 'miss_rate': miss_rate, 'f1': f1,
            'confusion': confusion}


def gen_confusion(test_results):
    """Generates a confusion matrix based on classifier test results.

    Args:
        test_results: Iterable of (true, pred) labels.

    Returns:
        Confusion matrix in the form conf[true_label][pred_label]
    """
    confusion = {}
    for true_label, pred_label in test_results:
        try:
            confusion[true_label][pred_label] += 1
        except KeyError:
            try:
                confusion[true_label][pred_label] = 1
            except KeyError:
                confusion[true_label] = {pred_label: 1}
    return confusion


def evaluate(classifier, label_values, class_selector=None, converted=False):
    """Classifies the provided values and generates stats based on the labels.

    Args:
        classifier: A classifier instance that conforms to the BinaryClassifier
            spec.
        label_values: Iterable of tuples of label and list-like objects.
            Example: [(label, value), ...]
        class_selector: Function that takes classifier output and returns a
            label. If None (default) then use first class label (highest
            confidence).
        converted: True then the input is in the correct internal format.

    Returns:
        A dictionary of performance statistics.
    """
    if class_selector == None:
        class_selector = lambda x: x[0][1]
    test_results = [(label, class_selector(classifier.predict(value, converted=converted)))
                    for label, value in label_values]
    confusion = gen_confusion(test_results)
    return confusion_stats(confusion)


def confidence_stats(classifier, label_values, samples=None):
    """Classifies the provided values and generates stats based on  the labels.

    Assumes labels are either -1 or 1.

    Args:
        classifier: A classifier instance that conforms to the BinaryClassifier
            spec.
        label_values: Iterable of tuples of label and list-like objects.
            Example: [(label, value), ...]
        samples: If None (default) then use every point.  Else select this many
            uniform samples.

    Returns:
        A dictionary where key is threshold and value is performance stats dict
    """
    conf = lambda x: x[0][0] * x[0][1]

    test_confidences = [(label, conf(classifier.predict(value)))
                        for label, value in label_values]
    confs = [x[1] for x in test_confidences]
    if samples == None:
        confs += [float('-inf'), float('inf')]
    else:
        min_conf = min(confs)
        max_conf = max(confs)
        step = (max_conf - min_conf) / float(samples)
        confs = np.arange(min_conf - step, max_conf + step, step).tolist()
    thresh_stats = {}
    for conf_thresh in confs:
        mkclass = lambda x: -1 if x < conf_thresh else 1
        test_results = ((x[0], mkclass(x[1])) for x in test_confidences)
        confusion = gen_confusion(test_results)
        thresh_stats[conf_thresh] = confusion_stats(confusion)
    return thresh_stats


def multi_evaluate(classifiers, label_values, class_selectors=None):
    """Classifies the provided values and generates stats based on  the labels.

    Args:
        classifiers: A list of classifiers that conforms to the BinaryClassifier spec.
        label_values: Iterable of tuples of label and list-like objects.
            Example: [(label, value), ...]
        class_selectors: List of functions (one per classifier, if less then
            use default) that take classifier output and return label. If None
            (default) then use first class label (highest confidence).

    Returns:
        A dictionary of performance statistics.
    """
    if class_selectors == None:
        class_selectors = [lambda x: x[0][1]]
    classifier_output = []
    for sel, cls in itertools.izip_longest(class_selectors, classifiers):
        classifier_output.append(evaluate(cls, label_values, sel))
    return {'classifier_output': classifier_output}


def hard_negatives(classifier, label_values, class_selector=None):
    """Classifies the provided values and generates stats based on  the labels.

    Args:
        classifier: A classifier instance that conforms to the BinaryClassifier
            spec.
        label_values: Iterable of tuples of label and list-like objects.
            Example: [(label, value), ...]
        class_selector: Function that takes classifier output and returns a
            label. If None (default) then use first class label (highest
            confidence).

    Returns:
        An iterator of values that correspond to false positives.
    """
    if class_selector == None:
        class_selector = lambda x: x[0][1]
    for label, value in label_values:
        if label != class_selector(classifier.predict(value)):
            yield value


def select_parameters(classifier_class, label_values, parameters, optimizer,
                      options=None, converted=False, num_folds=10):
    """Finds good parameters

    The optimizer must run in bounded time as we return its maximum value

    Args:
        classifier_class: A classifier that conforms to BinaryClassifier spec
        label_values: Iterable of tuples of label and list-like objects.
            Example: [(label, value), ...]
        parameters: Dict with keys as parameter names and values as
            (low, high, resolution) where generated parameters are [low, high)
            and resolution is a hint at the relevant scale of the parameter.
        optimizer: Function that takes (fitness_func, parameters) and returns
            an iterator of (fitness, params).  See Pyram Library for examples.
        options: Options to pass to the classifier unchanged (Default: None)
        converted: True then the input is in the correct internal format
        num_folds: Number of partitions to split the data into (default 10).
            If len(label_values) < num_folds, then we use len(label_values)

    Returns:
        (accuracy, params) where accuracy is the max value with associated params
    """
    if not options:
        options = {}
    if not converted:
        label_values = list(classifier_class.convert_label_values(label_values))

    def fitfunc(**kw):
        cur_options = dict(kw)
        cur_options.update(options)
        return cross_validation(classifier_class, label_values,
                                options=cur_options, converted=True,
                                num_folds=num_folds)
    vals = []
    for x in optimizer(fitfunc, parameters):
        vals.append(x)
    out_accuracy, out_params = max(vals)
    out_params.update(options)
    return out_accuracy, out_params


def whiten(label_values):
    """Convert values to zero mean and unit variance

    Values that have zero variance are replaced with 0 instead of
    NaN and warnings are suppressed.

    Args:
        label_values: Iterator of (label, value)

    Returns:
        List of label_values that have been scaled in each dimension to zero
        mean and unit variance
    """
    label_values = list(label_values)
    values = [x[1] for x in label_values]
    prev_err = np.seterr(all='ignore')
    value_mean = np.mean(values, 0)
    value_std = np.std(values, 0)
    out = [(x, np.nan_to_num((y - value_mean) / value_std)) for x, y in label_values]
    np.seterr(**prev_err)
    return out
