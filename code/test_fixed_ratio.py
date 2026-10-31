"""Regression tests for the fixed-ratio comparison."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('scipy', 'sklearn', 'lightgbm')),
                     'peer experiments require requirements-experiments.txt')
class FixedRatioTests(unittest.TestCase):
    def test_fixed_ratios_match_direct_formula(self):
        from submission_fixed_ratio import predict_scores
        from train_complement import relative_scores

        official = [['1', '2', None, '3'], ['4', '5', '6', '7'], [None] * 4]
        valid = np.array([[value is not None for value in row] for row in official])
        old = np.array([[0., 4., 9., 2.], [1., 2., 3., 4.], [0., 0., 0., 0.]])
        peer = np.array([[3., 1., 9., -2.], [4., 1., 3., 2.], [0., 0., 0., 0.]])
        baseline = relative_scores(old, valid)
        peer_relative = relative_scores(peer, valid)
        previous = 0.75 * baseline + 0.25 * peer_relative
        with (patch('submission_fixed_ratio.previous_scores', return_value=(previous, official)),
              patch('submission_fixed_ratio.normalized_scores', return_value=(old, official))):
            for weight in (0.15, 0.5):
                scores, parsed = predict_scores([{}, {}, {}], np.zeros((3, 4, 1)), {}, weight)
                np.testing.assert_allclose(scores, weight * baseline + (1 - weight) * peer_relative,
                                           atol=1e-14)
                self.assertEqual(parsed, official)

    def test_mismatched_official_answers_fail(self):
        from submission_fixed_ratio import predict_scores

        with (patch('submission_fixed_ratio.previous_scores',
                    return_value=(np.zeros((1, 4)), [['1'] * 4])),
              patch('submission_fixed_ratio.normalized_scores',
                    return_value=(np.zeros((1, 4)), [['2'] * 4]))):
            with self.assertRaisesRegex(ValueError, 'Official answer parsing'):
                predict_scores([{}], np.zeros((1, 4, 1)), {}, 0.5)

    def test_wrappers_fix_the_intended_weight(self):
        import submission_ratio_15_85
        import submission_ratio_50_50

        for module, weight in ((submission_ratio_15_85, 0.15), (submission_ratio_50_50, 0.5)):
            with patch.object(module, 'predict_fixed', return_value=[2]) as predict:
                self.assertEqual(module.predict([{}], {}, 'checkpoint'), [2])
                predict.assert_called_once_with([{}], {}, 'checkpoint', old_weight=weight)


if __name__ == '__main__':
    unittest.main()
