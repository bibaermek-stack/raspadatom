import unittest

import numpy as np

from spectra_ai.train_libs import split_samples


class SplitTests(unittest.TestCase):
    def test_identical_composition_stays_in_one_partition(self):
        chemistry = np.repeat(np.arange(32)[:, None], 3, axis=0)
        partitions = split_samples(chemistry, seed=7)
        membership = [set(chemistry[indices, 0]) for indices in partitions]
        self.assertEqual(sum(map(len, partitions)), len(chemistry))
        self.assertEqual(sum(map(len, membership)), 32)
        self.assertEqual(len(set.union(*membership)), 32)
        self.assertTrue(all(membership))
