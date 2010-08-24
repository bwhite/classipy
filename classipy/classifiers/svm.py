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
import numpy as np
import tempfile
import cPickle as pickle

import libsvm.svm
import libsvm.svmutil

from base import BinaryClassifier


class SVM(BinaryClassifier):
    def __init__(self, options=None):
        super(SVM, self).__init__()
        if not options:
            options = {}
        if 't' not in options:
            options['t'] = '0'
        try:
            self._param = ' '.join(['-%s %s' % x for x in options.items()])
        except AttributeError:
            self._param = ''
        self._param += ' -q'  # Makes silent
        self._m = None
        
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
        labels, values = zip(*list(label_values))
        prob  = libsvm.svm.svm_problem(labels, values)
        param = libsvm.svm.svm_parameter(self._param)
        self._m = libsvm.svmutil.svm_train(prob, param)
        return self

    def predict(self, value, converted=False):
        """Evaluates a single value against the training data.

        NOTE: Confidence is currently set to 0!

        Args:
            value: List-like object with same dimensionality used for training
                or the result of using convert_value if converted=True.
            converted: If True then the input is in the correct internal format.

        Returns:
            Sorted (descending) list of (confidence, label)
        """
        if not converted:
            value = self.convert_value(value)
        labels, stats, confidence = libsvm.svmutil.svm_predict([-1], [value], self._m)
        return [(math.fabs(confidence[0][0]), int(labels[0]))]

    @classmethod
    def convert_value(cls, value, *args, **kw):
        """Converts value to an efficient representation.

        Args:
            value: A value in a valid input type.

        Returns:
            Value in an efficient representation.
        """
        return super(SVM, cls).convert_value(value, to_type=list, *args, **kw)

    def dumps(self):
        """Serializes the classifier to a string

        Returns:
            A string that can be passed to the class' loads method
        """
        m = self._m
        self._m = None
        with tempfile.NamedTemporaryFile() as fp:
            libsvm.svmutil.svm_save_model(fp.name, m)
            fp.seek(0)
            ser_model = fp.read()
        out = pickle.dumps((self, ser_model))
        self._m = m
        return out
    
    @classmethod
    def loads(cls, s):
        """Returns a classifier instance given a serialized form

        Args:
            s: Serialized string

        Returns:
            An instance of this class as it was before it was serialized
        """
        c, ser_model = pickle.loads(s)
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(ser_model)
            fp.file.flush()
            m = libsvm.svmutil.svm_load_model(fp.name)
        c._m = m
        return c


def main():
    print(__doc__)

if __name__ == '__main__':
    main()
