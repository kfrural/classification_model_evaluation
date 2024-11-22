from sklearn.svm import SVC

class ClassificationModel:
    def __init__(self, kernel='linear', C=1, class_weight='auto'):
        self.model = SVC(kernel=kernel, C=C, class_weight=class_weight)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
