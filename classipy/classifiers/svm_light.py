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

try:
    import svmlight
except ImportError, e:
    print('Error: pysvmlight is not installed.  A copy is available in classipy/thirdparty')
    raise e
    
from base import BinaryClassifier


class SVMLight(BinaryClassifier):
    def __init__(self, options=None):
        super(SVMLight, self).__init__()

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
        self._m = svmlight.learn(list(label_values), type='classification',
                                 verbosity=1)
        return self

    @classmethod
    def convert_value(cls, value, *args, **kw):
        """Converts value to an efficient representation.

        Args:
            value: A value in a valid input type.

        Returns:
            Value in an efficient representation.
        """
        value = super(SVMLight, cls).convert_value(value, to_type=list, *args, **kw)
        return [(ind + 1, val) for ind, val in enumerate(value)]

    def predict(self, value, converted=False):
        """Evaluates a single value against the training data.

        Args:
            value: List-like object with same dimensionality used for training
                or the result of using convert_value if converted=True.
            converted: If True then the input is in the correct internal format.

        Returns:
            Sorted (descending) list of (confidence, label)
        """
        if not converted:
            value = self.convert_value(value)
        conf = svmlight.classify(self._m, [(0, value)])[0]
        return [(math.fabs(conf), cmp(conf, 0))]

def main():
    print(__doc__)

if __name__ == '__main__':
    main()
