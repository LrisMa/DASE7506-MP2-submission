"""Fixed 15% original / 85% peer comparison."""
from submission_fixed_ratio import predict_fixed


def predict(bank, arrays, checkpoint):
    return predict_fixed(bank, arrays, checkpoint, old_weight=0.15)
