import os
import numpy as np
import joblib
import preproc as pr
from evaluation import CrossValidation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Multiclass logistic regression (softmax) trained with gradient descent and L2 regularisation
def train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000, lambda_reg=2):
    m, n = X.shape
    theta = np.zeros((n, 3))  # 3 classes

    y_one_hot = np.eye(3)[y]

    for iteration in range(num_iterations):
        h = predict(X, theta)
        gradient = 1/m * X.T.dot(h - y_one_hot)

        # regularisation term (bias is not regularised)
        regularization_term = (lambda_reg / m) * theta[1:]

        theta[0] -= learning_rate * gradient[0]
        theta[1:] -= learning_rate * (gradient[1:] + regularization_term)

    return theta

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(X, theta):
    return softmax(np.dot(X, theta))

def add_bias(X):
    return np.c_[np.ones((len(X), 1)), X]


dataset_dir = os.path.join(BASE_DIR, "dataset")

images, labels, groups = pr.load_dataset(dataset_dir)
X = np.array([pr.extract_features(img) for img in images])
print(X.shape)

cv = CrossValidation('Logistic Regression')

for fold, (train_idx, test_idx) in enumerate(pr.get_folds(labels, groups), 1):
    theta = train_logistic_regression(add_bias(X[train_idx]), labels[train_idx])

    train_pred = np.argmax(predict(add_bias(X[train_idx]), theta), axis=1)
    test_pred = np.argmax(predict(add_bias(X[test_idx]), theta), axis=1)
    cv.add_fold(fold, labels[train_idx], train_pred, labels[test_idx], test_pred)

cv.summary(os.path.join(BASE_DIR, 'results'))

# final model trained on all the data, used by the app
theta_logistic = train_logistic_regression(add_bias(X), labels)
joblib.dump(theta_logistic, os.path.join(BASE_DIR, 'logisticregression_manual_weights.joblib'))
