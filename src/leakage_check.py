import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
import preproc as pr
from evaluation import load_features, metrics, RESULTS_DIR
from logistic_regression_manual import train, predict_labels

# Same model and features, two ways of splitting the data:
# - random split by image (what most papers on this dataset do)
# - split by patient group (slices of one patient are never in train and test at the same time)

X, labels, groups = load_features()

random_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, labels))
patient_folds = pr.get_folds(labels, groups)

results = {}
for name, folds in [('random split', random_folds), ('patient split', patient_folds)]:
    fold_results = [metrics(labels[te], predict_labels(train(X[tr], labels[tr]), X[te])) for tr, te in folds]
    results[name] = {k: float(np.mean([r[k] for r in fold_results])) for k in fold_results[0]}
    print(f"{name}: accuracy {results[name]['accuracy']:.4f} | macro F1 {results[name]['macro_f1']:.4f} | benign recall {results[name]['benign_recall']:.4f}")

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, 'leakage_check.json'), 'w') as f:
    json.dump(results, f, indent=2)
