import numpy as np


cdef partition(label_values, func):
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


cdef normalized_histogram(labels):
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


cdef entropy(q):
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


cdef information_gain(ql, qr):
    """Compute the information gain of the split

    Args:
        ql: List of (label, value)
        qr: List of (label, value)
    """
    h_q = entropy(ql + qr)
    h_ql = entropy(ql)
    h_qr = entropy(qr)
    pr_l = len(ql) / float(len(ql) + len(qr))
    pr_r = 1. - pr_l
    return h_q - pr_l * h_ql - pr_r * h_qr


cdef train_find_feature(label_values, num_feat, tree_depth, min_info,
                       make_feature):
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
        func = make_feature()
        ql, qr = partition(label_values, func)
        info_gain = information_gain(ql, qr)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_func = func
            max_info_gain_ql = ql
            max_info_gain_qr = qr
    if max_info_gain <= min_info:
        return [normalized_histogram(zip(*label_values)[0])]
    max_info_gain_func._max_info_gain = max_info_gain
    tree_depth = tree_depth - 1
    return (max_info_gain_func,
            train_find_feature(max_info_gain_ql, num_feat, tree_depth,
                               min_info, make_feature),
            train_find_feature(max_info_gain_qr, num_feat, tree_depth,
                               min_info, make_feature))


def train(label_values, make_feature, num_feat=10000, tree_depth=4,
          min_info=.1):
    return train_find_feature(label_values, num_feat, tree_depth, min_info, make_feature)
