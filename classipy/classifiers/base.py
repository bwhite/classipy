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
    
    def convert_value(self, value, to_type=None):
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

    def convert_label_values(self, label_values, to_type=None):
        """Converts an iterable of values to a list of to_type.

        Args:
	    label_values: Iterable of tuples of label and list-like objects.
                Example: [(label, value), ...]
            to_type: Overrides self.to_type (default None)

        Returns:
            An iterable of label_values in the type specified by to_type.
        """
        return ((label, self.convert_value(value, to_type)) for label, value in label_values)
