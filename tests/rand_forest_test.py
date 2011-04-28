import classipy
import unittest
import numpy as np

class Test(unittest.TestCase):

    def test_histogram_weights(self):
        num_classes = 50
        num_hists = 100
        num_samples = 100000
        labels = np.array(np.random.randint(0, num_classes, num_samples), dtype=np.int32)
        weights = np.array(np.random.randint(1, 100, num_samples * num_hists).reshape((num_hists, num_samples)), dtype=np.int32)
        out = classipy.rand_forest.histogram_weight(labels, weights, num_classes)
        self.assertEquals(out.shape, (num_hists, num_classes))

if __name__ == '__main__':
    unittest.main()

