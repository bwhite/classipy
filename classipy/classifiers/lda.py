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

"""Linear Descriminant Analysis
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'


import itertools
import numpy as np
from base import BinaryClassifier

def pca(data_matrix):
    """Computes the Principle Component Analysis on a data_matrix.

    Args:
        data_matrix: Each row is a data point, each column is a feature for
            all points.  E.g., (points, feature_dims)
    Returns:
        A tuple (projection, mean) where
        projection: numpy array with shape (points, feature_dims)
        mean: numpy array with shape (feature_dims)
    """
    data_matrix = np.array(data_matrix)
    mean = np.mean(data_matrix, 0)
    if data_matrix.shape[0] > data_matrix.shape[1]:
        return np.eye(data_matrix.shape[1]), mean
    data_matrix -= mean
    V = np.linalg.svd(data_matrix)[2]
    num_data = data_matrix.shape[0]
    return V[:num_data - 2], mean

def lda(data_matrix0, data_matrix1, prior0=.5):
    """Computes Fisher's LDA on 2 data matrices.

    Args:
        data_matrix0: Each row is a data point, each column is a feature for
            all points.  E.g., (points, feature_dims)
        data_matrix1: Each row is a data point, each column is a feature for
            all points.  E.g., (points, feature_dims)
    Returns:
        projection: numpy array with shape (feature_dims)
    """
    data_matrix0 = np.asmatrix(data_matrix0)
    data_matrix1 = np.asmatrix(data_matrix1)
    prior1 = 1. - prior0
    mu0 = np.mean(data_matrix0, 0)
    mu1 = np.mean(data_matrix1, 0)
    sw = prior0 * np.cov(data_matrix0.T) + prior1 * np.cov(data_matrix1.T)
    sw = np.asmatrix(sw)
    v = np.asarray(sw.I * (mu1 - mu0).T)
    # NOTE: This can happen when we are unable to compute variance (few pts)
    if np.isnan(v).all():
        return np.ones(v.shape)
    return np.nan_to_num(v / np.linalg.norm(v))

class LDA(BinaryClassifier):
    def __init__(self, options=None):
        super(LDA, self).__init__(options=options)
        try:
            self._prior0 = options['prior0']
        except (KeyError, TypeError):
            self._prior0 = .5

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
        data_dict = {-1: [], 1: []}
        # Input data is Points x Dims
        for label, value in label_values:
            data_dict[label].append(value)
        data_dict[-1] = np.array(data_dict[-1])
        data_dict[1] = np.array(data_dict[1])
        # Use PCA to project to Points x min(Points - 2, Dims)
        self._proj_pca, self._mean_pca = pca(np.concatenate((data_dict[-1], data_dict[1])))
        data_dict[-1] = np.dot(data_dict[-1] - self._mean_pca, self._proj_pca.T)
        data_dict[1] = np.dot(data_dict[1] - self._mean_pca, self._proj_pca.T)
        # Use LDA to project to Points x 1
        self._proj_lda = lda(data_dict[-1], data_dict[1], prior0=self._prior0)
        self._proj = np.dot(self._proj_lda.T, self._proj_pca)
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
        value = float(np.dot(value - self._mean_pca, self._proj.T))
        return [(float(np.abs(value)), cmp(value, 0))]

    @classmethod
    def convert_value(cls, value):
        """Converts value to an efficient representation.

        Args:
            value: A value in a valid input type.

        Returns:
            Value in an efficient representation.
        """
        return super(LDA, cls).convert_value(value, to_type=np.ndarray)

def main():
    print(__doc__)

if __name__ == '__main__':
    main()
