import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from evaluation import run_experiment, ROOT_DIR


# Gaussian Naive Bayes
def train_naive_bayes(X, y, class_priors=None):
    classes, class_counts = np.unique(y, return_counts=True)

    # priors can be set by hand for the imbalanced classes
    if class_priors is not None:
        class_probs = class_priors
    else:
        class_probs = class_counts / len(y)

    feature_probs = {}
    for c in classes:
        class_mask = (y == c)
        class_data = X[class_mask]
        feature_probs[c] = {
            'mean': np.mean(class_data, axis=0),
            'std': np.std(class_data, axis=0) + 1e-9  # avoid dividing by zero
        }

    return {'class_probs': class_probs, 'feature_probs': feature_probs, 'classes': classes}

def predict_naive_bayes(model_params, X):
    predictions = []
    for x in X:
        class_scores = []
        for c in model_params['classes']:
            class_prob = np.log(model_params['class_probs'][c])
            for i, feature_value in enumerate(x):
                mean = model_params['feature_probs'][c]['mean'][i]
                std = model_params['feature_probs'][c]['std'][i]
                log_likelihood = -0.5 * np.log(2 * np.pi * std**2) - ((feature_value - mean)**2) / (2 * std**2)
                class_prob += log_likelihood

            class_scores.append(class_prob)

        predicted_class = model_params['classes'][np.argmax(class_scores)]
        predictions.append(predicted_class)

    return np.array(predictions)


# PCA first, Naive Bayes assumes independent features and the PCA components are uncorrelated
def train(X, y):
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=20, random_state=42).fit(scaler.transform(X))
    nb = train_naive_bayes(pca.transform(scaler.transform(X)), y)
    return {'scaler': scaler, 'pca': pca, 'nb': nb}

def predict_labels(model, X):
    return predict_naive_bayes(model['nb'], model['pca'].transform(model['scaler'].transform(X)))


if __name__ == '__main__':
    final_model = run_experiment('Naive Bayes', train, predict_labels)
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    joblib.dump(final_model, os.path.join(ROOT_DIR, 'models', 'naive_bayes.joblib'))
