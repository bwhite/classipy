#import rand_forest
import _rand_forest as rand_forest
import random
import numpy as np


def hist_to_str(hist):
    hist = hist.items()
    hist.sort(key=lambda x: x[1], reverse=True)
    return ' '.join('%s:%.4g' % x for x in hist)


def feature_to_str(func):
    """Given a feature function, gives a string representation

    Args:
        func: Feature function

    Returns:
        String representation
    """
    return '%f <= x[%d]' % (func._thresh, func._dim)


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
        cur_name = '%s[label="I[%f]P[%s]"]' % (cur_id, tree[0]._max_info_gain,
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


def make_feature(dims):
    """Make a random decision feature on a vector

    Args:
        dims: Dimensions in the form [(min0, max0), ..., (min(N-1), max(N-1))]

    Returns:
        Function of the form func(vec) = Boolean, True iff the feature passes
    """
    dim = random.randint(0, len(dims) - 1)
    min_val, max_val = dims[dim]
    # [0, 1) -> [min_val, max_val)
    thresh = random.random() * (max_val - min_val) + min_val
    #print('[%d] x >= %f' % (dim, thresh))
    func = lambda vec: vec[dim] >= thresh
    func._dim = dim
    func._thresh = thresh
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
    label_values = data_generator(1000)
    dims = [(0., 1.), (0., 1.)]
    tree = rand_forest.train(label_values, lambda : make_feature(dims))
    print('\n\n')
    print(build_graphviz_tree(tree)[1])

if __name__ == '__main__':
    main()
