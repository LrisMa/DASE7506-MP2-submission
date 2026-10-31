"""Cheap, deterministic signals about candidate calculations and question quantities."""
import ast
from fractions import Fraction
import re

import numpy as np

NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
NUMBER_PATTERN = re.compile(rf'(?<![\w.])({NUMBER})(?:/({NUMBER}))?(%?)(?![\w.])')
FRACTION_PATTERN = re.compile(r'\\(?:d?frac)\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}')
EQUATION = re.compile(rf'(?<![\w.])((?:[-+]?{NUMBER})(?:\s*[+*/-]\s*[-+]?{NUMBER})+)\s*=\s*([-+]?{NUMBER})(?![\w.])')


def numbers(text):
    values = set()
    for match in NUMBER_PATTERN.finditer(text):
        numerator = Fraction(match.group(1).replace(',', ''))
        if match.group(2):
            denominator = Fraction(match.group(2).replace(',', ''))
            if denominator == 0:
                continue
            numerator /= denominator
        values.add(numerator)
        if match.group(3):
            values.add(numerator / 100)
    return values


def arithmetic_value(expression):
    """Evaluate only bounded numeric arithmetic; never execute arbitrary code."""
    if len(expression) > 100:
        raise ValueError('Arithmetic expression too long')
    expression = expression.replace(',', '')
    node = ast.parse(expression, mode='eval').body

    def evaluate(item):
        if isinstance(item, ast.Constant) and type(item.value) in (int, float):
            value = Fraction(str(item.value))
            if abs(value) > 10**12:
                raise ValueError('Operand too large')
            return value
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = evaluate(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and isinstance(item.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(item.left), evaluate(item.right)
            if isinstance(item.op, ast.Add):
                result = left + right
            elif isinstance(item.op, ast.Sub):
                result = left - right
            elif isinstance(item.op, ast.Mult):
                result = left * right
            else:
                result = left / right
            if abs(result.numerator) > 10**18 or result.denominator > 10**12:
                raise ValueError('Arithmetic result too large')
            return result
        raise ValueError('Unsupported arithmetic')

    return evaluate(node)


def normalized_math(text):
    text = FRACTION_PATTERN.sub(r'(\1/\2)', text)
    return (text.replace(r'\times', '*').replace(r'\cdot', '*')
            .replace(r'\div', '/').replace('×', '*').replace('÷', '/'))


def feature_row(question, text, answer):
    question_numbers = numbers(question)
    math = normalized_math(text)
    all_numbers = numbers(math)
    math_lines = '\n'.join(line for line in math.splitlines()
                           if any(symbol in line for symbol in ('=', '*', '/', '+', '-')))
    math_numbers = numbers(math_lines)
    checked = 0
    wrong = 0
    for match in EQUATION.finditer(math):
        try:
            left = arithmetic_value(match.group(1))
            right = Fraction(match.group(2).replace(',', ''))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        checked += 1
        wrong += left != right
    question_lower = question.lower()
    text_lower = text.lower()
    answer_value = Fraction(answer) if answer is not None else None
    units = (('week', 7), ('dozen', 12), ('hour', 60), ('day', 24), ('year', 12))
    unit_features = [float(unit in question_lower and Fraction(value) in math_numbers)
                     for unit, value in units]
    return [
        np.log1p(len(text)), np.log1p(text.count('\n')), np.log1p(text.count('=')),
        np.log1p(checked), np.log1p(wrong), float(wrong > 0),
        len(question_numbers & all_numbers) / max(1, len(question_numbers)),
        len(question_numbers & math_numbers) / max(1, len(question_numbers)),
        np.log1p(len(question_numbers - math_numbers)),
        float(answer_value is not None and answer_value in question_numbers),
        np.log1p(abs(float(answer_value))) if answer_value is not None and abs(answer_value) < 10**15 else 0,
        float('%' in question and ('%' in text or '0.' in text)),
        float('week' in question_lower and 'week' in text_lower),
        *unit_features,
    ]


def reasoning_features(bank, answers):
    out = np.zeros((len(bank), 4, 18), dtype=np.float32)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            out[i, j] = feature_row(row['question'], candidate['text'], answers[i][j])
    return out
