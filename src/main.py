import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from .models.model import ClassificationModel
from .utils.evaluation_metrics import calculate_metrics, print_classification_report, print_confusion_matrix

def main():
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ClassificationModel()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    print("Accuracy:", metrics['accuracy'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("F1 Score:", metrics['f1'])

    print("\nClassification Report:")
    print_classification_report(y_test, y_pred)

    print("\nConfusion Matrix:")
    print_confusion_matrix(y_test, y_pred)

    print("\nAdditional Metrics:")
    print("True Positives:", np.sum((y_test == 1) & (y_pred == 1)))
    print("False Positives:", np.sum((y_test == 0) & (y_pred == 1)))
    print("False Negatives:", np.sum((y_test == 1) & (y_pred == 0)))
    print("True Negatives:", np.sum((y_test == 0) & (y_pred == 0)))

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'AUC = {auc(fpr, tpr):0.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
