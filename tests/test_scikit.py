try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
import classipy

# Cheat Sheet (method/test) <http://docs.python.org/library/unittest.html>
#
# assertEqual(a, b)       a == b   
# assertNotEqual(a, b)    a != b    
# assertTrue(x)     bool(x) is True  
# assertFalse(x)    bool(x) is False  
# assertRaises(exc, fun, *args, **kwds) fun(*args, **kwds) raises exc
# assertAlmostEqual(a, b)  round(a-b, 7) == 0         
# assertNotAlmostEqual(a, b)          round(a-b, 7) != 0
# 
# Python 2.7+ (or using unittest2)
#
# assertIs(a, b)  a is b
# assertIsNot(a, b) a is not b
# assertIsNone(x)   x is None
# assertIsNotNone(x)  x is not None
# assertIn(a, b)      a in b
# assertNotIn(a, b)   a not in b
# assertIsInstance(a, b)    isinstance(a, b)
# assertNotIsInstance(a, b) not isinstance(a, b)
# assertRaisesRegexp(exc, re, fun, *args, **kwds) fun(*args, **kwds) raises exc and the message matches re
# assertGreater(a, b)       a > b
# assertGreaterEqual(a, b)  a >= b
# assertLess(a, b)      a < b
# assertLessEqual(a, b) a <= b
# assertRegexpMatches(s, re) regex.search(s)
# assertNotRegexpMatches(s, re)  not regex.search(s)
# assertItemsEqual(a, b)    sorted(a) == sorted(b) and works with unhashable objs
# assertDictContainsSubset(a, b)      all the key/value pairs in a exist in b

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_permute(self):
        # It isn't immediately clear how the decision
        # functions are ordered.  This shows that they are
        # almost certainly ordered by Y value as opposed to
        # when they first appear (as in stock libsvm)
        from scikits.learn import svm
        import random
        X = [[0, 0], [1, 1], [2, 2]]
        Y = [50, 51, 52]
        xy = zip(X, Y)
        for x in range(10):
            random.shuffle(xy)
            X, Y = zip(*xy)
            clf0 = svm.SVC()
            clf0.fit(X, Y)
            #print(clf0.decision_function([[2., 2.]]))
            #print(clf0.predict([[2., 2.]]))

    def test_histogram_intersection(self):
        label_values = [(0, [np.random.randint(30, 50), np.random.randint(0, 40)]) for x in range(50)]
        label_values += [(1, [np.random.randint(0, 40), np.random.randint(30, 50)]) for x in range(50)]
        
        def hik(x, y):
            return np.array([[np.sum(np.min([x0, y0], 0)) for x0 in x] for y0 in y])

        a = classipy.SVMScikit(kernel=hik).train(label_values)
        label_values = [(0, [np.random.randint(30, 50), np.random.randint(0, 40)]) for x in range(50)]
        label_values += [(1, [np.random.randint(0, 40), np.random.randint(30, 50)]) for x in range(50)]
        print(classipy.evaluate(a, label_values))

    def test_fast_hik(self):
        x, y = np.random.random((100, 50)), np.random.random((100, 50))

        def hik(x, y):
            return np.array([[np.sum(np.min([x0, y0], 0)) for y0 in y] for x0 in x])

        def my_kernel(x, y):
            return np.dot(x, y.T)

        import time
        st = time.time()
        out0 = classipy.kernels.histogram_intersection(x, y)
        print('HIK Fast[%f]' % (time.time() - st))
        st = time.time()
        out1 = hik(x, y)
        print('HIK[%f]' % (time.time() - st))
        self.assertEqual(my_kernel(x, y).shape, out0.shape)
        np.testing.assert_equal(out0, out1)

if __name__ == '__main__':
    unittest.main()
