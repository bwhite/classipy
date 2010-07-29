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

"""Fake Classifier for Stats
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np

from base import BinaryClassifier

class Fake(BinaryClassifier):
    def __init__(self, options=None):
        super(Fake, self).__init__()
        try:
            self._precision = options['precision']
        except (KeyError, TypeError):
            self._precision = .5
        try:
            self._recall = options['recall']
        except (KeyError, TypeError):
            self._recall = .5
        self._n = options['n']

    def train(self, *args, **kw):
        """Dummy method.

        Returns:
            self
        """
        return self

    def predict(self, value, *args, **kw):
        """Evaluates a single value against the training data.

        Assumes the first dimension of the value encodes the class (-1=Neg 1=Pos)
        Args:
            value: List-like object with same dimensionality used for training
                or the result of using convert_value if converted=True.

        Returns:
            Sorted (descending) list of (confidence, label).
        """
        gt_label = value[0]
        out_label = -1
        # p = tp / (tp + fp)
        # p = tp / total_pred_pos
        # total_pred_pos * p = tp
        # r = tp / (tp + fn) = tp / tot_pos
        # tot_pos * r = tp
        # tot_pos = tp + fn
        # tot_neg = tn + fp
        # tot_pos = tot_neg
        # P(pred=+ | gt=+) = TP / N = 
        # P(pred=+ | gt=-) = FP / N
        # P(pred=- | gt=+) = FN / N
        # P(pred=- | gt=-) = TN / N
        # Precision = 
        print(self._recall/self._precision - self._recall)
        if gt_label == 1:
            if np.random.random() < self._recall:
                out_label =  1 # TP
            else:
                out_label = -1 # FN
        else:
            if np.random.random() < 2/(.5*self._recall/self._precision - .5*self._recall):
                out_label = 1 # FP
            else:
                out_label = -1 # TN
                
        return [(.5, out_label)]


def main():
    print(__doc__)

if __name__ == '__main__':
    main()
