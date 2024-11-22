from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def calculate_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

def print_confusion_matrix(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
