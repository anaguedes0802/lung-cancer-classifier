import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
import preproc as pr
from evaluation import CrossValidation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

# PCA (fitted on the training data only) followed by Naive Bayes
def train_naive_bayes_pca(X, y, class_priors=None, n_components=None):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    nb_model_params = train_naive_bayes(X_pca, y, class_priors=class_priors)

    return {'nb_model_params': nb_model_params, 'pca': pca}

def predict_naive_bayes_pca(model_params, X):
    X_pca = model_params['pca'].transform(X)
    return predict_naive_bayes(model_params['nb_model_params'], X_pca)


dataset_dir = os.path.join(BASE_DIR, "dataset")

images, labels, groups = pr.load_dataset(dataset_dir)
X = np.array([pr.extract_features(img) for img in images])

n_components = 120

cv = CrossValidation('Naive Bayes + PCA')

for fold, (train_idx, test_idx) in enumerate(pr.get_folds(labels, groups), 1):
    model = train_naive_bayes_pca(X[train_idx], labels[train_idx], n_components=n_components)

    train_pred = predict_naive_bayes_pca(model, X[train_idx])
    test_pred = predict_naive_bayes_pca(model, X[test_idx])
    cv.add_fold(fold, labels[train_idx], train_pred, labels[test_idx], test_pred)

cv.summary(os.path.join(BASE_DIR, 'results'))

nb_model_params_with_pca = train_naive_bayes_pca(X, labels, n_components=n_components)
joblib.dump(nb_model_params_with_pca, os.path.join(BASE_DIR, 'naive_bayes_model_with_pca.joblib'))
