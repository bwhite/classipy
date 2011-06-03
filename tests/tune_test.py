import numpy as np
import pyram
import classipy

a = [(-1, x) for x in np.random.multivariate_normal([0, 0], np.eye(2), 1000)]
a += [(1, x) for x in np.random.multivariate_normal([3, 3], np.eye(2) * 2, 1000)]
b = classipy.select_parameters(classipy.SVM, a, {'c': (10**-1, 10**2, 10),
                                                 'g': (10**-1, 10**5, 10)},
                               pyram.exponential_grid, options={'t': '2'})
print(b)
