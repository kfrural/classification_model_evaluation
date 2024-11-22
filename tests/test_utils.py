import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.evaluation_metrics import calculate_metrics, print_classification_report, print_confusion_matrix

def test_calculate_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = calculate_metrics(y_true, y_pred)
    assert abs(metrics['accuracy'] - 0.75) < 0.01
    assert abs(metrics['precision'] - 0.67) < 0.01
    assert abs(metrics['recall'] - 0.5) < 0.01
    assert abs(metrics['f1'] - 0.58```
