import random
import numpy as np


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
        if len(tree) == 1:
            return [], []
        cur_id = str(graphviz_ctr[0])
        graphviz_ctr[0] += 1
        cur_name = '%s[label="I[%f]P[%s]"]' % (cur_id, tree[0]._max_info_gain, feature_to_str(tree[0]))
        node_names = [cur_name]
        links = []
        if parent:
            color = 'red' if left_node else 'green'
            links.append('%s->%s[color=%s]' % (parent, cur_id, color))

        def run_child(child_num):
            child_node_names, child_links = recurse(tree[child_num], parent=cur_id,
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


def partition(label_values, func):
    """
    Args:
        label_values: Iterator of vectors
        func: func(vec) = Boolean

    Returns:
        Tuple of (ql, qr)
        ql: Elements of vecs s.t. func is false
        qr: Elements of vecs s.t. func is true
    """
    ql, qr = [], []
    for label, value in label_values:
        if func(value):
            qr.append((label, value))
        else:
            ql.append((label, value))
    return ql, qr


def normalized_histogram(labels):
    """Computes a normalized histogram of labels

    Args:
        labels:  List of labels

    Returns:
        Dict with key's as unique values of labels, values as probabilities
    """
    out = {}
    if not labels:  # Nothing provided
        return {}
    for x in labels:
        try:
            out[x] += 1
        except KeyError:
            out[x] = 1
    norm = 1. / len(labels)
    # Ordered for numerical stability
    for x in out:
        out[x] = out[x] * norm
    return out


def entropy(q):
    """Shannon Entropy

    Args:
        q: List of (label, value)

    Returns:
        Entropy in 'bits'
    """
    try:
        q_labels, q_values = zip(*q)
    except ValueError:
        q_labels, q_values = [], []
    hist = normalized_histogram(q_labels).itervalues()
    return -sum(x * np.log2(x) for x in hist)


def information_gain(ql, qr):
    h_q = entropy(ql + qr)
    h_ql = entropy(ql)
    h_qr = entropy(qr)
    pr_l = len(ql) / float(len(ql) + len(qr))
    pr_r = 1. - pr_l
    return h_q - pr_l * h_ql - pr_r * h_qr


def train_find_feature(label_values, dims, num_feat, tree_depth, min_info):
    if tree_depth < 0:
        try:
            return [normalized_histogram(zip(*label_values)[0])]
        except IndexError:
            return {}
    max_info_gain = -float('inf')
    max_info_gain_func = None
    max_info_gain_ql = None
    max_info_gain_qr = None
    for feat_num in range(num_feat):
        func = make_feature(dims)
        ql, qr = partition(label_values, func)
        info_gain = information_gain(ql, qr)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_func = func
            max_info_gain_ql = ql
            max_info_gain_qr = qr
    tff = lambda x, y: train_find_feature(x, dims, num_feat, y,
                                          min_info)
    max_info_gain_func._max_info_gain = max_info_gain
    tree_depth = tree_depth - 1 if max_info_gain >= min_info else -1
    return (max_info_gain_func,
            tff(max_info_gain_ql, tree_depth),
            tff(max_info_gain_qr, tree_depth))


def train(label_values, dims, num_feat=1000, tree_depth=1, min_info=0):
    tree = train_find_feature(label_values, dims, num_feat, tree_depth, min_info)
    print('\n\n')
    print(build_graphviz_tree(tree)[1])
    #print('Best Feature[%d]:  x >= %f' % (f._dim, f._thresh))


def data_generator(num_points):
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
    train(label_values, dims)
    #val = make_feature([(0, 1), (3, 5)])
    #print(val([.5, 4.]))

if __name__ == '__main__':
    main()
