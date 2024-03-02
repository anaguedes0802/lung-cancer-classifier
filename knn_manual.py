import os
import numpy as np
from collections import Counter
import joblib
import preproc as pr
from evaluation import CrossValidation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_knn(X_train, y_train, k=3):
    # KNN has no real training, it just keeps the training data
    return {
        'X_train': X_train,
        'y_train': y_train,
        'k': k
    }

def predict_knn(model_params, sample):
    distances = np.linalg.norm(model_params['X_train'] - sample, axis=1)
    nearest_indices = np.argsort(distances)[:model_params['k']]
    nearest_labels = model_params['y_train'][nearest_indices]

    # majority class among the k nearest neighbours
    most_common = Counter(nearest_labels).most_common(1)
    predicted_label = most_common[0][0]

    return predicted_label


dataset_dir = os.path.join(BASE_DIR, 'dataset')

images, labels, groups = pr.load_dataset(dataset_dir)
X = np.array([pr.extract_features(img) for img in images])

cv = CrossValidation('KNN')

for fold, (train_idx, test_idx) in enumerate(pr.get_folds(labels, groups), 1):
    knn_model_params = train_knn(X[train_idx], labels[train_idx], k=3)

    # on the training set each sample is its own nearest neighbour, so this one is optimistic
    train_pred = [predict_knn(knn_model_params, sample) for sample in X[train_idx]]
    test_pred = [predict_knn(knn_model_params, sample) for sample in X[test_idx]]
    cv.add_fold(fold, labels[train_idx], train_pred, labels[test_idx], test_pred)

cv.summary(os.path.join(BASE_DIR, 'results'))

knn_model_params = train_knn(X, labels, k=3)
joblib.dump(knn_model_params, os.path.join(BASE_DIR, 'knn_model.joblib'))
