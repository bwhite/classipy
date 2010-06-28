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


class BinaryClassifier(object):
    def __init__(self, options=None):
    	"""Initializes classifier

        Args:
            options: A dictionary of options specific to the classifier.
        """
        super(BinaryClassifier, self).__init__()
        self.to_type = list  #  Either numpy.ndarray, tuple, or list.
    
    def _convert_value(self, value, to_type=None):
        """Converts value to to_type.

        Args:
            value: A value in a valid to_type.
            to_type: Overrides self.to_type (default None)

        Returns:
            Value in the type specified by to_type.
        """
        if to_type == None:
            to_type = self.to_type
        if isinstance(value, to_type):  # Same type, quit early
            return value
        if to_type == np.ndarray: # If it needs to be numpy
            return np.array(value)
        if isinstance(value, np.ndarray): # We know to_type isn't numpy
            value = value.tolist()
        if to_type == tuple:
            return tuple(value)
        if to_type == list:
            return list(value)

    def _convert_values(self, values, to_type=None):
        """Converts an iterator of values to a list of to_type.

        Args:
            values: A list of values (must be homogeneous)
            to_type: Overrides self.to_type (default None)

        Returns:
            A list of values in the type specified by to_type.
        """
        # TODO It would be nicer if numpy returned a 2D numpy array
        return [self._convert_value(value, to_type) for value in values]

    def _train_normalize(self, values):
        """Learns the min/max for a series of values, setting _shift, _scale.

        Args:
            values: List of list-like objects, all with the same dimensionality.
        Returns:
            Normalized values.
        """
        values = self._convert_values(values)
        values = np.array(values)
        mi = np.min(values, 0)
        ma = np.max(values, 0)
        self._shift = mi
        self._scale = 1 / (ma - mi)
        self._scale[np.isinf(self._scale)] = 0.  # Force inf to zero
        return [self._normalize(x) for x in values]

    def _normalize(self, value):
        """Normalizes a value [0, 1]

        Args:
            values: List-like object.
        Returns:
            Normalized value
        """
        value = self._convert_value(value, to_type=np.ndarray)
        value -= self._shift
        value *= self._scale
        return self._convert_value(value)
