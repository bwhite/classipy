import numpy as np
cimport numpy as np


cdef extern from "kernels_aux.h":
    void histogram_intersection_fast(double *x, double *y, int rows_x, int rows_y, int cols, double *out)

cpdef histogram_intersection(np.ndarray[np.float64_t, ndim=2, mode='c'] x, np.ndarray[np.float64_t, ndim=2, mode='c'] y):
    cdef np.ndarray out = np.zeros((x.shape[0], y.shape[0]))
    histogram_intersection_fast(<double *>x.data, <double *>y.data, x.shape[0], y.shape[0], x.shape[1], <double *>out.data)
    return out
    #return np.array([[np.sum(np.min([x0, y0], 0)) for x0 in x] for y0 in y])
