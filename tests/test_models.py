from unittest.mock import patch
from src.models.model import ClassificationModel

def test_model_init():
    model = ClassificationModel()
    assert isinstance(model.model, object)

@patch('sklearn.svm.SVC')
def test_model_fit(mock_svc):
    X = [[1, 2], [3, 4]]
    y = [0, 1]
    model = ClassificationModel()
    model.fit(X, y)
    mock_svc.fit.assert_called_once_with(X, y)

@patch('sklearn.svc.SVC.predict')
def test_model_predict(mock_predict, model):
    X = [[1, 2], [3, 4]]
    model.predict = mock_predict
    result = model.predict(X)
    mock_predict.assert_called_once_with(X)
