import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from evaluation import run_experiment, ROOT_DIR


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(input_size, hidden_sizes, output_size):
    sizes = [input_size] + hidden_sizes + [output_size]
    weights = [np.random.randn(sizes[i], sizes[i + 1]) / np.sqrt(sizes[i]) for i in range(len(sizes) - 1)]
    return weights

def forward_pass(inputs, weights):
    layer_outputs = [inputs]
    for w in weights[:-1]:
        layer_inputs = np.dot(layer_outputs[-1], w)
        layer_outputs.append(sigmoid(layer_inputs))

    # output layer without activation (softmax is applied later)
    output_inputs = np.dot(layer_outputs[-1], weights[-1])
    return layer_outputs, output_inputs

def backward_pass(layer_outputs, weights, error):
    gradients = [error]

    for i in range(len(weights) - 1, 0, -1):
        gradient = gradients[-1].dot(weights[i].T) * layer_outputs[i] * (1 - layer_outputs[i])
        gradients.append(gradient)

    gradients.reverse()
    return gradients

def train_step(inputs, labels_one_hot, weights, learning_rate):
    layer_outputs, output_inputs = forward_pass(inputs, weights)

    # softmax on the output layer
    output = np.exp(output_inputs - np.max(output_inputs, axis=1, keepdims=True))
    output /= np.sum(output, axis=1, keepdims=True)

    error = labels_one_hot - output
    gradients = backward_pass(layer_outputs, weights, error)

    for i in range(len(weights)):
        weights[i] += learning_rate * layer_outputs[i].T.dot(gradients[i])

    return weights

def train_neural_network(X, y, hidden_sizes=[128, 64], learning_rate=0.001, epochs=50, batch_size=32):
    output_size = 3
    weights = initialize_weights(X.shape[1], hidden_sizes, output_size)

    for epoch in range(epochs):
        # shuffle the training set every epoch
        order = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = order[i:i + batch_size]
            batch_labels = np.eye(output_size)[y[batch_idx]]
            weights = train_step(X[batch_idx], batch_labels, weights, learning_rate)

    return weights

def predict(inputs, weights):
    _, output_inputs = forward_pass(inputs, weights)
    output = np.exp(output_inputs - np.max(output_inputs, axis=1, keepdims=True))
    output /= np.sum(output, axis=1, keepdims=True)
    return np.argmax(output, axis=1)


def train(X, y):
    np.random.seed(42)
    scaler = StandardScaler().fit(X)
    weights = train_neural_network(scaler.transform(X), y)
    return {'scaler': scaler, 'weights': weights}

def predict_labels(model, X):
    return predict(model['scaler'].transform(X), model['weights'])


if __name__ == '__main__':
    final_model = run_experiment('Neural Network', train, predict_labels)
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    joblib.dump(final_model, os.path.join(ROOT_DIR, 'models', 'neural_network.joblib'))
