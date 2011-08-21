"""
Usage:

python rand_forest_run.py (task)
where task is one of
    test: Generates a classifier and tests it (default)
    profile: Runs test in profile mode (for best results compile rand_forest.pyx
        with the profiling comment)
"""
import classipy
import random
import numpy as np
import cPickle as pickle
import sys


def data_generator(num_points):
    """
    Args:
        num_points: Number of points to generate
    """
    # Here we make a few fake classes and see if the classifier can get it
    cgens = [[(.2, .4), (0, 1)], [(.3, .6), (0, 1)], [(.3, .6), (0, 1)]]
    print(cgens)
    out = []
    for x in range(num_points):
        label = random.randint(0, len(cgens) - 1)
        value = [np.random.uniform(x, y) for x, y in cgens[label]]
        if label == 2:
            value.append(label)
        else:
            value.append(0)
        out.append((label, value))
    return out


def train():
    label_values = data_generator(50000)
    dims = np.array([(0., 1.), (0., 1.), (0., 3.)])
    feature_factory = classipy.rand_forest.VectorFeatureFactory(dims, np.array([0, 0, 2]), 10)
    rfc = classipy.RandomForestClassifier(feature_factory,
                                          num_feat=100)
    rfc.train(label_values)
    return rfc, label_values, dims


def main():
    rfc, label_values, dims = train()
    print('Predicting')
    correct = 0
    total = 0
    for x, y in label_values:
        total += 1
        if rfc.predict(y)[0][1] == x:
            correct += 1
    print('%f/%f' % (correct, total))
    print('\n\n')
    print('Decision tree graphs (open in your browser)')
    print(rfc.graphviz_google())


def prof():
    import pstats
    import cProfile
    cProfile.runctx("main()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    s.strip_dirs().sort_stats("cum").print_stats()

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        main()
    elif sys.argv[1] == 'profile':
        prof()
    elif sys.argv[1] == 'time_train':
        from timeit import Timer
        t = Timer("train()", "from __main__ import train")
        print t.timeit(number=20)
    else:
        print(__doc__)
