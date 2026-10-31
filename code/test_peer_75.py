"""Tests for reweighting the frozen peer-inspired ensemble."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('scipy', 'sklearn', 'lightgbm')),
                     'peer experiments require requirements-experiments.txt')
class PeerBlendTests(unittest.TestCase):
    def test_reweights_standardized_components_and_masks(self):
        from submission_peer_75 import predict_scores
        from train_complement import relative_scores

        official = [['1', '2', None, '3'], ['4', '5', '6', '7'], [None] * 4]
        valid = np.array([[value is not None for value in row] for row in official])
        old = np.array([[0., 4., 9., 2.], [1., 2., 3., 4.], [0., 0., 0., 0.]])
        peer = np.array([[3., 1., 9., -2.], [4., 1., 3., 2.], [0., 0., 0., 0.]])
        baseline = relative_scores(old, valid)
        peer_relative = relative_scores(peer, valid)
        previous = 0.75 * baseline + 0.25 * peer_relative
        with (patch('submission_peer_75.previous_scores', return_value=(previous, official)),
              patch('submission_peer_75.normalized_scores', return_value=(old, official))):
            scores, parsed = predict_scores([{}, {}, {}], np.zeros((3, 4, 1)), {})
        np.testing.assert_allclose(scores, 0.25 * baseline + 0.75 * peer_relative, atol=1e-14)
        self.assertEqual(parsed, official)

    def test_rejects_mismatched_official_answers(self):
        from submission_peer_75 import predict_scores

        with (patch('submission_peer_75.previous_scores', return_value=(np.zeros((1, 4)), [['1'] * 4])),
              patch('submission_peer_75.normalized_scores', return_value=(np.zeros((1, 4)), [['2'] * 4]))):
            with self.assertRaisesRegex(ValueError, 'Official answer parsing'):
                predict_scores([{}], np.zeros((1, 4, 1)), {})

    def test_empty_bank_and_missing_checkpoint(self):
        from submission_peer_75 import predict

        self.assertEqual(predict([], {}, Path('unused.npz')), [])
        with self.assertRaisesRegex(ValueError, 'requires a checkpoint'):
            predict([], {}, None)

    def test_predict_masks_unparseable_candidates_and_falls_back(self):
        from submission_peer_75 import predict

        scores = np.array([[10., 2., 1., 0.], [0., 0., 0., 0.]])
        official = [[None, '2', '3', '4'], [None] * 4]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.npz'
            np.savez(checkpoint, example=np.array([1]))
            with patch('submission_peer_75.predict_scores', return_value=(scores, official)):
                self.assertEqual(predict([{}, {}], {'embeddings': np.zeros((2, 4, 1))}, checkpoint),
                                 [1, 0])


if __name__ == '__main__':
    unittest.main()
