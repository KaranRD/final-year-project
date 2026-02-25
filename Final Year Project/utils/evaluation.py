def accuracy(y_true, y_pred):
    """Calculate the accuracy between true labels and predicted labels."""
    return sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred):
    """Calculate precision score."""
    true_positive = sum((y_true == 1) & (y_pred == 1))
    false_positive = sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0


def recall(y_true, y_pred):
    """Calculate recall score."""
    true_positive = sum((y_true == 1) & (y_pred == 1))
    false_negative = sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0


def f1_score(y_true, y_pred):
    """Calculate F1 score."""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

