"""Conservative parenthesized-equation diagnostics using training data only."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import time

import numpy as np

from common import load_inputs, score_choices
from reasoning_features import arithmetic_value, feature_row, normalized_math
from submission import candidate_features, select
from submission_normalized import normalize_bank
from train_complement import normalized_scores
from train_experiments import evaluate, labels_for
from train_improved import hard_mask


CHARS = r'0-9.,+*/()\-\t '
PAREN_EQUATION = re.compile(
    rf'(?<![\w.])(?P<left>[{CHARS}]*\([{CHARS}]+\)[{CHARS}]*)[ \t]*='
    rf'[ \t]*(?P<right>[{CHARS}]+)(?![\w.%^/0-9.,+*()\-\t ])'
)
NUMBER_TOKEN = re.compile(r'\d[\d,]*(?:\.\d+)?')
GROUPED_NUMBER = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?')
DECIMAL = re.compile(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)\.(\d+)')
PERCENT_SUFFIX = re.compile(r'(?:\\?%|percent\b|per[ \t]+cent\b|\\text\{[ \t]*percent\b)', re.IGNORECASE)
REMAINDER_SUFFIX = re.compile(r'(?:with[ \t]+a[ \t]+remainder\b|\\text\{[ \t]*with[ \t]+a[ \t]+remainder\b|\\(?:l?dots)\b)', re.IGNORECASE)


def invalid_grouping(expression):
    return any(',' in token and not GROUPED_NUMBER.fullmatch(token)
               for token in NUMBER_TOKEN.findall(expression))


def parenthesized_equation_checks(text):
    """Return whether each conservatively recognized equation is unequal."""
    math = normalized_math(text)
    checks = []
    for match in PAREN_EQUATION.finditer(math):
        left, right = match.group('left').strip(), match.group('right').strip()
        prefix = math[:match.start('left')].rstrip(' \t')
        before_delimiter = prefix.rstrip('$').rstrip(' \t')
        if (left.startswith('+') or
                (left.startswith('-') and prefix and (prefix[-1].isalnum() or prefix[-1] in '}%)')) or
                (before_delimiter and before_delimiter[-1] in '+-*/') or
                re.search(r'\d[ \t]*[xX][ \t]*$', before_delimiter)):
            continue
        suffix = math[match.end('right'):].lstrip(' \t')
        if PERCENT_SUFFIX.match(suffix) or REMAINDER_SUFFIX.match(suffix):
            continue
        if invalid_grouping(left) or invalid_grouping(right):
            continue
        try:
            left_value = arithmetic_value(left)
            right_value = arithmetic_value(right)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        decimal = DECIMAL.fullmatch(right)
        tolerance = Fraction(1, 10 ** len(decimal.group(1))) if decimal else Fraction(0)
        checks.append(abs(left_value - right_value) > tolerance)
    return checks


def equation_features(normalized_bank):
    features = np.zeros((len(normalized_bank), 4, 3), dtype=np.float32)
    for i, row in enumerate(normalized_bank):
        if len(row['candidates']) != 4:
            raise ValueError('Expected four candidates per question')
        for j, candidate in enumerate(row['candidates']):
            checks = parenthesized_equation_checks(candidate['text'])
            wrong = sum(checks)
            features[i, j] = [np.log1p(len(checks)), np.log1p(wrong), float(wrong > 0)]
    return features


def numeric_vocabulary_coverage(normalized_bank, question_vocabulary, candidate_vocabulary):
    from train_peer import tokens

    counts = {
        'validation_questions': len(normalized_bank), 'questions_with_numeric_tokens': 0,
        'questions_with_unseen_numeric_tokens': 0, 'question_numeric_tokens': 0,
        'question_unseen_numeric_tokens': 0, 'candidates': 0,
        'candidates_with_numeric_tokens': 0, 'candidates_with_unseen_numeric_tokens': 0,
        'candidate_numeric_tokens': 0, 'candidate_unseen_numeric_tokens': 0,
    }
    for row in normalized_bank:
        question = [token for token in tokens(row['question']) if re.search(r'\d', token)]
        missing = [token for token in question if token not in question_vocabulary]
        counts['questions_with_numeric_tokens'] += bool(question)
        counts['questions_with_unseen_numeric_tokens'] += bool(missing)
        counts['question_numeric_tokens'] += len(question)
        counts['question_unseen_numeric_tokens'] += len(missing)
        for candidate in row['candidates']:
            numbers = [token for token in tokens(candidate['text']) if re.search(r'\d', token)]
            missing = [token for token in numbers if token not in candidate_vocabulary]
            counts['candidates'] += 1
            counts['candidates_with_numeric_tokens'] += bool(numbers)
            counts['candidates_with_unseen_numeric_tokens'] += bool(missing)
            counts['candidate_numeric_tokens'] += len(numbers)
            counts['candidate_unseen_numeric_tokens'] += len(missing)
    return counts


def validation_audit():
    from submission_peer_75 import predict_scores

    bank, arrays = load_inputs('validation')
    with np.load('runs/peer-test/peer-bottom.npz', allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    blended, official = predict_scores(bank, arrays['embeddings'], model)
    old_model = {key[4:]: value for key, value in model.items() if key.startswith('old_')}
    baseline, old_official = normalized_scores(bank, arrays['embeddings'], old_model)
    if official != old_official:
        raise ValueError('Official answer parsing differs between ensemble branches')
    selections = {'baseline': select(baseline, official), 'peer75': select(blended, official)}
    normalized = normalize_bank(bank)
    flagged = equation_features(normalized)[:, :, 2] > 0
    question_vocabulary = json.loads(model['question_vocabulary'].tobytes().decode('utf-8'))
    candidate_vocabulary = json.loads(model['solution_vocabulary'].tobytes().decode('utf-8'))
    vocabulary = numeric_vocabulary_coverage(normalized, question_vocabulary, candidate_vocabulary)
    labels = labels_for(bank, 'validation')
    hard = hard_mask(official)
    result = {}
    for name, choices in selections.items():
        metrics = evaluate(choices, labels, official)
        official_metrics = score_choices('validation', [row['id'] for row in bank], choices)
        if any(metrics[key] != official_metrics[key] for key in ('correct', 'n', 'accuracy')):
            raise ValueError(f'Official validation score mismatch: {name}')
        chosen = np.asarray(choices)
        correct = labels[np.arange(len(bank)), chosen] > 0
        chosen_flagged = flagged[np.arange(len(bank)), chosen] & hard
        result[name] = {**metrics,
                        'disagreement_correct_selected_flagged': int((chosen_flagged & correct).sum()),
                        'disagreement_wrong_selected_flagged': int((chosen_flagged & ~correct).sum())}
    baseline_choice = np.asarray(selections['baseline'])
    peer_choice = np.asarray(selections['peer75'])
    baseline_correct = labels[np.arange(len(bank)), baseline_choice] > 0
    peer_correct = labels[np.arange(len(bank)), peer_choice] > 0
    different = np.array([official[i][baseline_choice[i]] != official[i][peer_choice[i]]
                          for i in range(len(bank))])
    paired = {
        'disagreement_questions': int(hard.sum()),
        'different_final_answers': int((hard & different).sum()),
        'baseline_only_correct': int((hard & baseline_correct & ~peer_correct).sum()),
        'peer75_only_correct': int((hard & ~baseline_correct & peer_correct).sum()),
        'both_wrong': int((hard & ~baseline_correct & ~peer_correct).sum()),
    }
    return {
        'data': 'validation inputs and references; choices computed before references were loaded',
        'caveat': ('Validation was previously reused for development and is not independent. '
                   'A flagged equation is not proof that its final answer is wrong. '
                   'No model is modified or trained by this audit.'),
        'results': result, 'paired_disagreement': paired,
        'numeric_vocabulary_coverage': vocabulary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('runs/normalization/equation-validation-summary.json' if args.validation
                           else 'runs/normalization/equation-audit.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump({'status': 'starting'}, out)
        out.write('\n')
    started = time.perf_counter()
    if args.validation:
        summary = validation_audit()
        summary['status'] = 'complete'
        summary['elapsed_seconds'] = time.perf_counter() - started
        args.output.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        return
    bank, _ = load_inputs('train')
    labels = labels_for(bank, 'train')
    official, _, _ = candidate_features(bank)
    hard = hard_mask(official)
    normalized = normalize_bank(bank)
    counts = {
        'train_questions': len(bank), 'disagreement_questions': int(hard.sum()),
        'questions_with_checked_equations': 0, 'disagreement_questions_with_checked_equations': 0,
        'candidates_with_checked_equations': 0, 'checked_equations': 0, 'unequal_equations': 0,
        'correct_candidates_checked': 0, 'incorrect_candidates_checked': 0,
        'correct_candidates_with_unequal_equation': 0, 'incorrect_candidates_with_unequal_equation': 0,
        'correct_candidates_new_unequal': 0, 'incorrect_candidates_new_unequal': 0,
    }
    for i, row in enumerate(normalized):
        found = False
        for j, candidate in enumerate(row['candidates']):
            checks = parenthesized_equation_checks(candidate['text'])
            if not checks:
                continue
            found = True
            counts['candidates_with_checked_equations'] += 1
            counts['checked_equations'] += len(checks)
            counts['unequal_equations'] += sum(checks)
            correct = labels[i, j] > 0
            prefix = 'correct' if correct else 'incorrect'
            counts[f'{prefix}_candidates_checked'] += 1
            if any(checks):
                counts[f'{prefix}_candidates_with_unequal_equation'] += 1
                old_wrong = feature_row(row['question'], candidate['text'], official[i][j])[5] > 0
                if not old_wrong:
                    counts[f'{prefix}_candidates_new_unequal'] += 1
        if found:
            counts['questions_with_checked_equations'] += 1
            counts['disagreement_questions_with_checked_equations'] += int(hard[i])
    summary = {
        'status': 'complete',
        'data': 'training split only; normalized question and candidate texts',
        'caveat': ('Conservative syntactic checks can miss equations and still misinterpret context. '
                   'Candidate counts are correlated; unequal arithmetic does not imply a wrong final answer. '
                   'No model is modified or trained by this audit.'),
        'counts': counts,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
