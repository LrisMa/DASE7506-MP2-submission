"""Tests for conservative training-only equation diagnostics."""
import importlib.util
import unittest

import numpy as np

from submission_normalized import normalize_bank
from test_submission import bank_row
from train_equation_diagnostics import (equation_features, numeric_vocabulary_coverage,
                                        parenthesized_equation_checks)


class EquationDiagnosticsTests(unittest.TestCase):
    def test_complete_equations_and_chains(self):
        checks = parenthesized_equation_checks
        self.assertEqual(checks('(2 + 3) * 4 = 20'), [False])
        self.assertEqual(checks('(2 + 3) * 4 = 19'), [True])
        self.assertEqual(checks('18 * (1 + 0.5) = 18 * 1.5 = 27'), [False])
        self.assertEqual(checks('(1/4) + (3/4) = 1'), [False])
        self.assertEqual(checks('(-2 + 5) * 4 = 12'), [False])

    def test_rounded_decimals_are_not_called_unequal(self):
        checks = parenthesized_equation_checks
        self.assertEqual(checks('(25 / 60) = 0.4167'), [False])
        self.assertEqual(checks('(25 / 60) = 0.4166'), [False])
        self.assertEqual(checks('(25 / 60) = 0.6'), [True])

    def test_unsupported_context_is_skipped(self):
        checks = parenthesized_equation_checks
        self.assertEqual(checks('12 + y + (12 + 2) = 50'), [])
        self.assertEqual(checks('4 * $ (3/4) = 3'), [])
        self.assertEqual(checks(r'(1/4) = 25\%'), [])
        self.assertEqual(checks('(1/4) = 25 percent'), [])
        self.assertEqual(checks(r'100\% - (1/3) = (2/3)'), [])
        self.assertEqual(checks(r'(49/3) = 16 \text{ with a remainder of } 1'), [])
        self.assertEqual(checks(r'(132/22.5) = 5.999\ldots'), [])
        self.assertEqual(checks('(1 / 0) = 0'), [])
        self.assertEqual(checks('(1,23 + 4) = 127'), [])
        self.assertEqual(checks('(2 ** 3) = 8'), [])

    def test_features_use_candidate_equations_not_question_numbers(self):
        row = bank_row(['(two + three) * 4 = 20', '(2 + 3) * 4 = 19',
                        '(25 / 60) = 0.4167', '(1 / 0) = 0'])
        row['question'] = 'Two apples and nine oranges.'
        normalized = normalize_bank([row])
        first = equation_features(normalized)
        self.assertEqual(first.shape, (1, 4, 3))
        np.testing.assert_allclose(first[0, 0], [np.log(2), 0, 0])
        np.testing.assert_allclose(first[0, 1], [np.log(2), np.log(2), 1])
        np.testing.assert_allclose(first[0, 2], [np.log(2), 0, 0])
        np.testing.assert_allclose(first[0, 3], [0, 0, 0])
        normalized[0]['question'] = 'Different values: 500 and 700.'
        np.testing.assert_array_equal(equation_features(normalized), first)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('scipy', 'sklearn', 'lightgbm')),
                         'peer experiments require requirements-experiments.txt')
    def test_numeric_vocabulary_coverage_uses_normalized_input_tokens(self):
        row = bank_row(['#### 2', '#### 2', '#### 7', '#### 9'])
        row['question'] = 'Two apples and nine oranges.'
        result = numeric_vocabulary_coverage(normalize_bank([row]), {'2': 0}, {'2': 0, '7': 1})
        self.assertEqual(result['questions_with_unseen_numeric_tokens'], 1)
        self.assertEqual(result['question_unseen_numeric_tokens'], 1)
        self.assertEqual(result['candidates_with_unseen_numeric_tokens'], 1)
        self.assertEqual(result['candidate_unseen_numeric_tokens'], 1)


if __name__ == '__main__':
    unittest.main()
