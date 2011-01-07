import numpy as np
import mp
import classipy

np.random.seed(5)
a = [(-1, x) for x in np.random.multivariate_normal([0, 0],
                                                    np.random.random((2, 2)) * 5, 1000)]
a += [(1, x) for x in np.random.multivariate_normal([3, 3],
                                                    np.random.random((2, 2)) * 5, 1000)]
for x in range(-5, 5):
    o = {'c': 1., 'g': 10**x, 't': 2}
    classifier = classipy.SVM(options=o).train(a)
    classipy.decision_boundary_2d(classifier, a, .5, .5)
    mp.title(str(o))

for x in range(-5, 5):
    o = {'c': 1., 'g': 10**x, 't': 2}
    classifier = classipy.SVM(options={'c': 10**x, 'g': 1, 't': 2}).train(a)
    classipy.decision_boundary_2d(classifier, a, .5, .5)
    mp.title(str(o))
