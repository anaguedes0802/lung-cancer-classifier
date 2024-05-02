import os
import numpy as np
from collections import Counter
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from evaluation import run_experiment, ROOT_DIR


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


# distances in 4096 dimensions are not very meaningful, so we use 50 PCA components
def train(X, y):
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=50, random_state=42).fit(scaler.transform(X))
    knn = train_knn(pca.transform(scaler.transform(X)), y, k=3)
    return {'scaler': scaler, 'pca': pca, 'knn': knn}

def predict_labels(model, X):
    Z = model['pca'].transform(model['scaler'].transform(X))
    return np.array([predict_knn(model['knn'], z) for z in Z])


if __name__ == '__main__':
    final_model = run_experiment('KNN', train, predict_labels)
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    joblib.dump(final_model, os.path.join(ROOT_DIR, 'models', 'knn.joblib'))
