import unittest
import zlib

import numpy as np

from spectra_ai.pretrain_measured import deflate_rows, partition, select_channels


class MeasuredDataTests(unittest.TestCase):
    def test_gaps_are_unavailable(self):
        columns, valid = select_channels(np.array([240., 241., 350.]), np.array([241., 300., 350.]))
        np.testing.assert_array_equal(columns, [1, 2])
        np.testing.assert_array_equal(valid, [True, False, True])

    def test_target_partition_is_stable(self):
        result = [partition(f"target{i}") for i in range(100)]
        self.assertEqual(set(result), {0, 1, 2})
        self.assertEqual(result, [partition(f"target{i}") for i in range(100)])

    def test_deflate_preserves_selected_rows(self):
        data = np.arange(60, dtype="<f8").reshape(12, 5)
        rows = deflate_rows(zlib.compress(data.tobytes()), data.dtype, 5, [1, 3, 11], block_size=13)
        np.testing.assert_array_equal(rows, data[[1, 3, 11]].T)
