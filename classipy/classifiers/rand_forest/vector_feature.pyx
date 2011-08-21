import numpy as np
cimport numpy as np


cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] normalized_histogram(np.ndarray[np.int32_t, ndim=1, mode='c'] labels, int num_classes):
    """Computes a histogram of labels

    Args:
        labels:  Ndarray of labels (ints) (must be 0 <= x < num_classes)

    Returns:
        Ndarray of length 'num_classes' with indexes as labels and
        values as probs
    """
    cdef np.ndarray out = histogram(labels, num_classes)
    cdef double scale = 1./ np.sum(out)
    return scale * out


# This is a sample FeatureFactory that implicitly specifies the interface
# This will be kept up to date to work with the script in the examples dir.
# It uses the feature below this.
cdef class VectorFeatureFactory(object):
    """
    Args:
        dims: Numpy array of min_val, max_val where [min_val, max_val) shape of (num_dims, 2)
        types: Numpy array where 0: Real, 1: Integer, 2: Categorical shape of (num_dims).  Real
            generates continuous values and uses a <= feature, integer generates integral values
            and uses a <= feature, and categorical generates integral values and uses a != feature.
        num_thresh: Number of thresholds per feature
        
    """
    cdef np.ndarray dims, types
    cdef int num_thresh

    def __init__(self, dims=None, types=None, num_thresh=None, label_values=None):
        self.dims = np.asarray(dims) if dims is not None else dims
        self.num_thresh = num_thresh if num_thresh is not None else 0
        self.types = np.asarray(types) if types is not None else types
        if label_values:
            values = [x[1] for x in label_values]
            self.dims = np.dstack([np.min(values, 0), np.max(values, 0)])[0]

    def gen_feature(self):
        cdef int dim = random.randint(0, len(self.dims) - 1)
        min_val, max_val = self.dims[dim]
        cdef int feat_type = self.types[dim]
        if feat_type == 0:
            threshs = np.random.uniform(min_val, max_val, self.num_thresh)
        elif feat_type == 1 or feat_type == 2:
            threshs = np.random.randint(min_val, max_val, self.num_thresh)
        else:
            raise ValueError('Feature type not recognized')
        threshs = np.ascontiguousarray(threshs.reshape((threshs.size, 1)))
        return VectorFeature(dim=dim, threshs=threshs, feat_type=feat_type)

    def loads(self, feat_ser):
        return VectorFeature(feat_ser=feat_ser)

    def select_feature(self, feats, feat_ind):
        """Select a feature by index

        This is used because each feature may have many internal configurations

        Args:
            feats: List of features
            feat_ind: Integer feature index

        Return:
            Feature
        """
        return feats[feat_ind / self.num_thresh][feat_ind % self.num_thresh]

    def leaf_probability(self, labels, values, num_classes):
        return normalized_histogram(labels, num_classes)


cdef class VectorFeature(object):
    cdef object feat_ser
    cdef int dim
    cdef np.ndarray threshs
    cdef int feat_type

    def __init__(self, feat_ser=None, dim=None, threshs=None, feat_type=None):
        self.feat_ser = feat_ser
        if self.feat_ser:
            self._deserialize()
        else:
            self.dim = dim
            self.threshs = threshs
            self.feat_type = feat_type

    def _deserialize(self):
        data = pickle.loads(self.feat_ser)
        self.dim = data['dim']
        self.threshs = data['threshs']
        self.feat_type = data['feat_type']

    def __str__(self):
        if self.threshs.size == 1:
            t = self.threshs[0][0]
        else:
            t = self.threshs
        if self.feat_type < 2:
            o = '<='
        else:
            o = '=='
        return '%s %s x[%d]' % (t, o, self.dim)

    def dumps(self):
        return pickle.dumps({'dim': self.dim, 'threshs': self.threshs, 'feat_type': self.feat_type}, -1)

    def __repr__(self):
        return 'VectorFeature(dim=%r, threshs=%r, feat_type=%f)' % (self.dim, self.threshs, self.feat_type)

    def __getitem__(self, index):
        return VectorFeature(dim=self.dim,
                             feat_type=self.feat_type,
                             threshs=np.array([[self.threshs.flat[int(index)]]]))

    def __call__(self, values):
        """
        Args:
            values: Values of the prespecified form
        
        Returns:
            Boolean array where neg/pos_inds are of shape (num_thresh, num_values)
        """
        values = np.asarray(values)
        if values.ndim == 1:
            v = values[self.dim]
        else:
            v = values[:, self.dim]
        if self.feat_type < 2:
            return v >= self.threshs
        else:
            return v == self.threshs

    def label_histograms(self, labels, values, int num_classes):
        """
        Args:
            labels: np.array of ints
            values: np.array of vectors

        Returns:
            Tuple of (qls, qrs)
            qls: Histograms of left labels with shape (num_thresh, num_classses)
            qrs: Histograms of right labels with shape (num_thresh, num_classses)
        """
        values = np.asarray(values)
        qls, qrs = [], []
        for x in self(values):
            qls.append(histogram(labels[~x], num_classes))
            qrs.append(histogram(labels[x], num_classes))
        return np.vstack(qls), np.vstack(qrs)

    def label_values_partition(self, labels, values):
        """Only uses the first row of values, producing 1 partition

        Args:
            labels: Iterator of ints
            values: Iterator of vecs

        Returns:
            Tuple of (ql_lab, ql_val, qr_lab, qr_val)
            ql: Elements of vecs s.t. func is false
            qr: Elements of vecs s.t. func is true
        """
        values = np.asarray(values)
        ql_lab, qr_lab = [], []
        ql_val, qr_val = [], []
        x = self(values)[0]
        return labels[~x], values[~x], labels[x], values[x]
