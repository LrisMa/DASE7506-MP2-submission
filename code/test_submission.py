"""Focused tests for answer grouping, selection and model inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import subprocess
import sys

import numpy as np

from submission import candidate_features, model_scores, predict, select
from train_experiments import evaluate, fit_binary, fit_pairwise, labels_for, standardize


def bank_row(texts):
    return {'id': 'unused', 'question': 'unused',
            'candidates': [{'index': i, 'text': text} for i, text in enumerate(texts)]}


class SubmissionTests(unittest.TestCase):
    def test_training_label_loader_rejects_test_and_audit_before_reading(self):
        with patch('train_experiments.read_jsonl', return_value=[]) as read:
            for split in ('test', 'audit'):
                with self.assertRaisesRegex(ValueError, 'limited to train and validation'):
                    labels_for([], split)
            read.assert_not_called()
            self.assertEqual(labels_for([], 'train').shape, (0, 4))
            self.assertEqual(labels_for([], 'validation').shape, (0, 4))
            self.assertEqual(read.call_count, 2)

    def test_numeric_equivalence_and_ties(self):
        bank = [bank_row([r'\boxed{1/2}', r'\boxed{0.5}', r'\boxed{1}', 'not parseable']),
                bank_row([r'\boxed{1}', r'\boxed{2}', r'\boxed{2}', r'\boxed{1}'])]
        answers, votes, _ = candidate_features(bank)
        self.assertEqual(answers[0], ['1/2', '1/2', '1', None])
        self.assertEqual(votes.tolist(), [[2, 2, 1, 0], [2, 2, 2, 2]])
        self.assertEqual(select(votes, answers), [0, 0])
        self.assertEqual(predict(bank, {'embeddings': np.zeros((2, 4, 1024))}, None), [0, 0])

    def test_no_parseable_answers_and_empty_bank(self):
        bank = [bank_row(['a', 'b', 'c', 'd'])]
        self.assertEqual(predict(bank, {'embeddings': np.zeros((1, 4, 1024))}, None), [0])
        self.assertEqual(predict([], {'embeddings': np.zeros((0, 4, 1024))}, None), [])

    def test_invalid_candidate_is_never_chosen_when_valid_exists(self):
        answers = [[None, '1', '2', '3']]
        self.assertEqual(select(np.array([[100, 1, 0, 0]]), answers), [1])

    def test_checkpoint_round_trip_and_variable_bank_size(self):
        bank = [bank_row([r'\boxed{1}', r'\boxed{2}', r'\boxed{3}', r'\boxed{4}'])]
        emb = np.zeros((1, 4, 1024), dtype=np.float32)
        emb[0, 2, 0] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            weights = np.zeros((1024, 1), dtype=np.float32)
            weights[0, 0] = 1
            np.savez(path, kind=np.array('linear'), mean=np.zeros(1024),
                     scale=np.ones(1024), w=weights, b=np.zeros(1), vote_weight=0)
            self.assertEqual(predict(bank, {'embeddings': emb}, path), [2])

    def test_fitting_and_evaluation(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(8, 4, 3)).astype(np.float32)
        labels = (x[:, :, 0] > 0).astype(np.float32)
        labels[0] = [1, 0, 1, 0]
        train, val, mean, scale = standardize(x, x)
        for kind in ('linear', 'mlp'):
            model = {'kind': np.array(kind), 'mean': mean, 'scale': scale,
                     **fit_binary(train, labels, kind, 123, 2, 8)}
            self.assertEqual(model_scores(x, np.zeros((8, 4, 5)), model).shape, (8, 4))
        pair = fit_pairwise(train, labels, 123, 2, 4)
        self.assertTrue(np.isfinite(pair['w']).all())
        self.assertEqual(val.shape, train.shape)
        result = evaluate([0] * 8, labels, [['1', '2', '3', '4']] * 8)
        self.assertEqual(result['n'], 8)
        self.assertEqual(result['disagreement_n'], 8)

    def test_audit_runner_without_reference_answers(self):
        project = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / 'candidates').mkdir()
            bank = [bank_row([r'\boxed{1}', r'\boxed{2}', r'\boxed{2}', r'\boxed{3}'])]
            bank_file = data / 'candidates' / 'gsm8k_audit.jsonl'
            array_file = data / 'candidates' / 'gsm8k_audit.npz'
            bank_file.write_text(json.dumps(bank[0]) + '\n')
            np.savez(array_file, embeddings=np.zeros((4, 1024), dtype=np.float32))
            hashes = {str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (bank_file, array_file)}
            (data / 'manifest.json').write_text(json.dumps({
                'protocol': 'dase7506-mp2-open-hard-verifier-v5',
                'splits': {'audit': 1}, 'sha256': hashes}))
            output = data / 'predictions.json'
            subprocess.run([sys.executable, str(project / 'predict.py'), '--method',
                            str(project / 'submission.py'), '--split', 'audit',
                            '--data-dir', str(data), '--output', str(output)],
                           check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertEqual(result['ids'], ['unused'])
            self.assertEqual(result['selected'], [1])
            checkpoint = data / 'model.npz'
            np.savez(checkpoint, kind=np.array('pairwise'), mean=np.zeros(1024),
                     scale=np.ones(1024), w=np.zeros((1024, 1)), b=np.zeros(1),
                     vote_weight=0.5)
            subprocess.run([sys.executable, str(project / 'predict.py'), '--method',
                            str(project / 'submission.py'), '--checkpoint', str(checkpoint),
                            '--split', 'audit', '--data-dir', str(data), '--output', str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())['selected'], [1])


if __name__ == '__main__':
    unittest.main()
