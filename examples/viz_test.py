import classipy
import numpy as np
import matplotlib.pyplot as mp

np.random.seed(5)
a = [(-1, x) for x in np.random.multivariate_normal([0, 0],
                                                    np.random.random((2, 2)) * 5, 1000)]
a += [(1, x) for x in np.random.multivariate_normal([3, 3],
                                                    np.random.random((2, 2)) * 5, 1000)]


def classifiers():
    for x in dir(classipy):
        classifier = getattr(classipy, x)
        try:
            if issubclass(classifier, classipy.BinaryClassifier):
                if classifier != classipy.BinaryClassifier:
                    yield x, classifier
        except TypeError:
            pass

for name, classifier in classifiers():
    mp.title(name)
    c = classifier().train(a)
    classipy.decision_boundary_2d(c, a, .5, .5)
