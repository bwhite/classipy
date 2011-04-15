#import rand_forest
import _rand_forest as rand_forest
import random
import numpy as np
import cPickle as pickle
import glob
import copy


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


def gen_feature(max_dist=76041):
    """Make a random decision feature on a vector

    To set max_dist select a depth image for your task.  Find the distance
    between 2 points that represent your maximum feature size roughly at the
    same depth in pixels, multiply this by their depth.

    Args:
        max_dist: Maximum distance in pixels * depth units

    Returns:
        Serialialized string (opaque)
    """
    rand_uni = lambda : np.random.uniform(-max_dist, max_dist)
    u = np.array([rand_uni(), rand_uni()])
    v = np.array([rand_uni(), rand_uni()])
    thresh = np.random.uniform(-2**8, 2**8)  # TODO Tweak this distribution
    return pickle.dumps({'u': u, 'v': v, 't': thresh})


def depth_samp(depth_image, x):
    big_number = 2**11 - 1  # TODO Tweak
    if min(x) < 0:
        return big_number
    try:
        return int(min(big_number, depth_image[x[0], x[1]]))
    except IndexError:
        return big_number


def make_feature_func(feat_str):
    """Load a feature form a serialized string

    Args:
        feat_str: Serialized feature string from gen_feature

    Returns:
        Function of the form func(vec) = Boolean, True iff the feature passes
    """
    data = pickle.loads(feat_str)
    u = data['u']
    v = data['v']
    t = data['t']

    def func(dx):
        d, x = dx
        d_x_inv = 1. / depth_samp(d, x)
        outa = (depth_samp(d, x + u * d_x_inv) -
                depth_samp(d, x + v * d_x_inv))
        #print(outa)
        out = (outa >= t)
        #print('[%s][%s][%s] : %f - %f >= %f: %s' % (x, x + u * d_x_inv, x + v * d_x_inv, depth_samp(d, x + u * d_x_inv), depth_samp(d, x + v * d_x_inv), t, out))
        return out
    func.__u = u
    func.__v = v
    func.__t = t
    return func


def data_generator(points_per_class):
    """
    Args:
        num_points: Number of points to generate
    """
    out = []
    depth_points = [np.array([x, y]) for x in range(480) for y in range(640)]
    for label, data_path in [(0, './wally0/'), (1, './notwally1/')]:
        samples = []
        images = glob.glob(data_path + '/*.pgm')
        points_per_image = int(np.ceil(points_per_class / float(len(images))))
        for image_path in images:
            with open(image_path) as fp:
                depth = fp.read().split('\n', 1)[1]
                depth = np.fromstring(depth, dtype=np.uint16)
                depth = depth.reshape((480, 640))
            cur_samples = random.sample(depth_points,
                                        points_per_image)
            samples += [(label, (depth, x)) for x in cur_samples]
        out += random.sample(samples, points_per_class)
    return out


def main():
    label_values = data_generator(10000)
    test_label_values = data_generator(10000)
    rfc = rand_forest.RandomForestClassifier(make_feature_func,
                                             gen_feature)
    rfc.train(label_values)
    correct = 0
    total = 0
    for x, y in test_label_values:
        total += 1
        if rfc.predict(y)[0][1] == x:
            correct += 1
    print('%f/%f' % (correct, total))
    print('\n\n')
    #print(build_graphviz_tree(rfc.tree)[1])


#def prof():
#    import pstats
#    import cProfile
#    label_values = data_generator(1000)
#    dims = [(0., 1.), (0., 1.)]
#    aa = lambda : make_feature(dims)
#    cProfile.runctx("rand_forest.train(label_values, aa, 2)", globals(), locals(), "Profile.prof")
#    s = pstats.Stats("Profile.prof")
#    s.strip_dirs().sort_stats("time").print_stats()


if __name__ == '__main__':
    main()
    #prof()
