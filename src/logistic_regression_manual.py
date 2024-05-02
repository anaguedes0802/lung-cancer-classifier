import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from evaluation import run_experiment, ROOT_DIR


# Multiclass logistic regression (softmax) trained with gradient descent and L2 regularisation.
# Each sample is weighted by the inverse frequency of its class, so the few benign
# images count as much as the many malignant ones.
def train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000, lambda_reg=20):
    m, n = X.shape
    theta = np.zeros((n, 3))  # 3 classes

    y_one_hot = np.eye(3)[y]
    class_weights = len(y) / (3 * np.bincount(y, minlength=3))
    sample_weights = class_weights[y][:, None]

    for iteration in range(num_iterations):
        h = predict(X, theta)
        gradient = 1/m * X.T.dot((h - y_one_hot) * sample_weights)

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


def train(X, y):
    scaler = StandardScaler().fit(X)
    theta = train_logistic_regression(add_bias(scaler.transform(X)), y)
    return {'scaler': scaler, 'theta': theta}

def predict_labels(model, X):
    return np.argmax(predict(add_bias(model['scaler'].transform(X)), model['theta']), axis=1)


if __name__ == '__main__':
    final_model = run_experiment('Logistic Regression', train, predict_labels)
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    joblib.dump(final_model, os.path.join(ROOT_DIR, 'models', 'logistic_regression.joblib'))
