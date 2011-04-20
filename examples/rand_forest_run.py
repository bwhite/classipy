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


def hist_to_str(hist):
    hist = list(enumerate(hist))
    hist.sort(key=lambda x: x[1], reverse=True)
    return ' '.join('%s:%.4g' % x for x in hist)


def feature_to_str(func):
    """Given a feature function, gives a string representation

    Args:
        func: Feature function

    Returns:
        String representation
    """
    return '%f <= x[%d]' % (func.__thresh, func.__dim)


def build_graphviz_tree(tree):
    graphviz_ctr = [0]

    def recurse(tree, parent='', left_node=False):
        cur_id = str(graphviz_ctr[0])
        graphviz_ctr[0] += 1
        color = 'red' if left_node else 'green'
        if len(tree) == 1:  # Leaf
            cur_name = '%s[label="%s"]' % (cur_id, hist_to_str(tree[0]))
            node_names, links = [cur_name], []
            links.append('%s->%s[color=%s]' % (parent, cur_id, color))
            return node_names, links
        cur_name = '%s[label="I[%f]P[%s]"]' % (cur_id, tree[3]['info_gain'],
                                               feature_to_str(tree[0]))
        node_names = [cur_name]
        links = []
        if parent:
            links.append('%s->%s[color=%s]' % (parent, cur_id, color))

        def run_child(child_num):
            child_node_names, child_links = recurse(tree[child_num],
                                                    parent=cur_id,
                                                    left_node=child_num == 1)
            node_names.extend(child_node_names)
            links.extend(child_links)
        run_child(1)
        run_child(2)
        return node_names, links
    node_names, links = recurse(tree)
    gv = 'digraph{%s}' % ';'.join(node_names + links)
    google_gv = 'https://chart.googleapis.com/chart?cht=gv:dot&chl=%s' % gv
    return gv, google_gv


def gen_feature(dims):
    """Make a random decision feature on a vector

    Args:
        dims: Dimensions in the form [(min0, max0), ..., (min(N-1), max(N-1))]

    Returns:
        Serialialized string (opaque)
    """
    dim = random.randint(0, len(dims) - 1)
    min_val, max_val = dims[dim]
    # [0, 1) -> [min_val, max_val)
    thresh = random.random() * (max_val - min_val) + min_val
    return pickle.dumps({'dim': dim, 'thresh': thresh})


def make_feature_func(feat_str):
    """Load a feature form a serialized string

    Args:
        feat_str: Serialized feature string from gen_feature

    Returns:
        Function of the form func(vec) = Boolean, True iff the feature passes
    """
    data = pickle.loads(feat_str)
    dim = data['dim']
    thresh = data['thresh']
    func = lambda vec: vec[dim] >= thresh
    func.__dim = dim
    func.__thresh = thresh
    return func


def data_generator(num_points):
    """
    Args:
        num_points: Number of points to generate
    """
    # Here we make a few fake classes and see if the classifier can get it
    cgens = [[(.2, .4), (0, 1)], [(.3, .6), (0, 1)]]
    print(cgens)
    out = []
    for x in range(num_points):
        label = random.randint(0, len(cgens) - 1)
        value = [np.random.uniform(x, y) for x, y in cgens[label]]
        out.append((label, value))
    return out


def main():
    label_values = data_generator(50000)
    dims = [(0., 1.), (0., 1.)]
    rfc = classipy.RandomForestClassifier(make_feature_func,
                                          lambda : gen_feature(dims),
                                          num_trees=1,
                                          num_procs=8,
                                          num_feat=1000)
    rfc.train(label_values)
    # Test pickle
    print('Pickling')
    rfc = classipy.RandomForestClassifier.loads(rfc.dumps(), make_feature_func,
                                                lambda : gen_feature(dims))
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
    for t in rfc.trees:
        print(build_graphviz_tree(t)[1])
        print('')


def prof():
    import pstats
    import cProfile
    cProfile.runctx("main()", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        main()
    elif sys.argv[1] == 'profile':
        prof()
    else:
        print(__doc__)
