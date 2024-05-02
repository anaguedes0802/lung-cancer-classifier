import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import preproc as pr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

CLASS_NAMES = ['Benign', 'Malignant', 'Normal']


def load_features():
    data = np.load(os.path.join(ROOT_DIR, 'features', 'features.npz'))
    return data['X'], data['labels'], data['groups']


def metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'benign_recall': recall_score(y_true, y_pred, labels=[0], average='macro'),
        'malignant_recall': recall_score(y_true, y_pred, labels=[1], average='macro'),
    }


def plot_confusion_matrix(conf_matrix, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()


def run_experiment(model_name, train_fn, predict_fn):
    # 1) 5-fold cross-validation (by patient) on the development set
    # 2) train on the whole development set and evaluate once on the held-out patients
    # 3) train on everything for the app
    X, labels, groups = load_features()
    dev, test = pr.dev_test_split(labels, groups)
    X_dev, y_dev, g_dev = X[dev], labels[dev], groups[dev]

    print(f'{model_name}')
    print(f'Development set: {len(dev)} images, test set: {len(test)} images ({len(np.unique(groups[test]))} patient groups)\n')

    fold_results = []
    for fold, (tr, te) in enumerate(pr.get_folds(y_dev, g_dev), 1):
        model = train_fn(X_dev[tr], y_dev[tr])
        m = metrics(y_dev[te], predict_fn(model, X_dev[te]))
        fold_results.append(m)
        print(f"Fold {fold}: accuracy {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f} | benign recall {m['benign_recall']:.4f}")

    cv_summary = {k: [float(np.mean([r[k] for r in fold_results])), float(np.std([r[k] for r in fold_results]))] for k in fold_results[0]}
    print('\nCross-validation on the development set (mean +- std):')
    for k, (mean, std) in cv_summary.items():
        print(f'  {k}: {mean:.4f} +- {std:.4f}')

    model = train_fn(X_dev, y_dev)
    test_pred = predict_fn(model, X[test])
    test_metrics = metrics(labels[test], test_pred)

    print('\nHeld-out test set:')
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}')
    print(classification_report(labels[test], test_pred, target_names=CLASS_NAMES, digits=4))
    conf_matrix = confusion_matrix(labels[test], test_pred)
    print(conf_matrix)

    slug = model_name.lower().replace(' ', '_')
    plot_confusion_matrix(conf_matrix, f'{model_name} (held-out test)', f'confusion_matrix_{slug}.png')
    with open(os.path.join(RESULTS_DIR, f'metrics_{slug}.json'), 'w') as f:
        json.dump({'cross_validation': cv_summary, 'test': test_metrics, 'confusion_matrix': conf_matrix.tolist()}, f, indent=2)

    return train_fn(X, labels)
