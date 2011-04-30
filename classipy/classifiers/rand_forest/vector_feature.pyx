import numpy as np
cimport numpy as np

# This is a sample FeatureFactory that implicitly specifies the interface
# This will be kept up to date to work with the script in the examples dir.
# It uses the feature below this.
cdef class VectorFeatureFactory(object):
    cdef object dims
    cdef int num_thresh

    def __init__(self, dims, num_thresh):
        self.dims = dims
        self.num_thresh = num_thresh

    def gen_feature(self):
        dim = random.randint(0, len(self.dims) - 1)
        min_val, max_val = self.dims[dim]
        threshs = np.array([np.random.uniform(min_val, max_val,
                                              self.num_thresh)]).T
        return VectorFeature(dim=dim, threshs=threshs)

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


cdef class VectorFeature(object):
    cdef object feat_ser
    cdef int dim
    cdef threshs

    def __init__(self, feat_ser=None, dim=None, threshs=None):
        self.feat_ser = feat_ser
        if self.feat_ser:
            self._deserialize()
        else:
            self.dim = dim
            self.threshs = threshs

    def _deserialize(self):
        data = pickle.loads(self.feat_ser)
        self.dim = data['dim']
        self.threshs = data['threshs']

    def __str__(self):
        if self.threshs.size == 1:
            return '%s <= x[%d]' % (self.threshs[0][0], self.dim)
        return '%s <= x[%d]' % (self.threshs, self.dim)

    def dumps(self):
        return pickle.dumps({'dim': self.dim, 'threshs': self.threshs}, -1)

    def __repr__(self):
        return 'VectorFeature(dim=%r, threshs=%r)' % (self.dim, self.threshs)

    def __getitem__(self, index):
        return VectorFeature(dim=self.dim,
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
            return values[self.dim] >= self.threshs
        else:
            return values[:, self.dim] >= self.threshs

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
