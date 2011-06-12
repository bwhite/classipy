try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
import classipy


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_whiten(self):
        values = (np.random.random((50, 10)) * 5) + 3
        label_values = [(0, x) for x in values]
        label_values_white = classipy.whiten(label_values)
        label_values_white2 = classipy.whiten(label_values_white)
        np.testing.assert_almost_equal([x[1] for x in label_values_white],
                                       [x[1] for x in label_values_white2])
        self.assertEqual([x[0] for x in label_values_white],
                         [x[0] for x in label_values_white2])
        values = [x[1] for x in label_values_white]
        np.testing.assert_almost_equal(np.std(values, 0), np.ones(len(values[0])))
        np.testing.assert_almost_equal(np.mean(values, 0), np.zeros(len(values[0])))
        print(values)

if __name__ == '__main__':
    unittest.main()
