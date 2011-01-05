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
import liblinear.linear
import liblinear.linearutil
from base import BinaryClassifier


class SVMLinear(BinaryClassifier):

    def __init__(self, options=None):
        super(SVMLinear, self).__init__()
        self._predict_param = ''
        try:
            if 'b' in options:
                self._predict_param= '-b %s' % (options['b'])
                del options['b']
            self._param = ' '.join(['-%s %s' % x for x in options.items()])
        except (AttributeError, TypeError):
            self._param = ''
        self._param += ' -q'  # Makes silent
        self._m = None

    def train(self, label_values, converted=False):
        """Build a model.

        Args:
            label_values: Iterable of tuples of label and list-like objects
                Example: [(label, value), ...]
                or the result of using convert_label_values if converted=True.
            converted: If True then the input is in the correct internal format
        Returns:
            self
        """
        if not converted:
            label_values = self.convert_label_values(label_values)
        labels, values = zip(*list(label_values))
        prob = liblinear.linear.problem(labels, values, pregen=True)
        param = liblinear.linear.parameter(self._param)
        self._m = liblinear.linearutil.train(prob, param)
        return self

    def predict(self, value, converted=False):
        """Evaluates a single value against the training data.

        Args:
            value: List-like object with same dimensionality used for training
                or the result of using convert_value if converted=True.
            converted: If True then the input is in the correct internal format

        Returns:
            Sorted (descending) list of (confidence, label)
        """
        if not converted:
            value = self.convert_value(value)
        opt = self._predict_param
        labels, stats, confidence = liblinear.linearutil.predict([-1],
                                                                 [value],
                                                                 self._m,
                                                                 options=opt)
        return [(math.fabs(confidence[0][0]), labels[0])]

    @classmethod
    def convert_value(cls, value):
        """Converts value to an efficient representation.

        Args:
            value: A value in a valid input type.

        Returns:
            Value in an efficient representation.
        """
        value = super(SVMLinear, cls).convert_value(value, to_type=list)
        return liblinear.linear.gen_feature_nodearray(value, issparse=False)[0]

    def dumps(self):
        """Serializes the classifier to a string

        Returns:
            A string that can be passed to the class' loads method
        """
        m = self._m
        self._m = None
        with tempfile.NamedTemporaryFile() as fp:
            liblinear.linearutil.save_model(fp.name, m)
            fp.seek(0)
            ser_model = fp.read()
        out = pickle.dumps((self, ser_model), -1)
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
            m = liblinear.linearutil.load_model(fp.name)
        c._m = m
        return c


def main():
    print(__doc__)

if __name__ == '__main__':
    main()
