import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ['Benign', 'Malignant', 'Normal']


class CrossValidation:
    # keeps the predictions of every fold and prints the summary at the end

    def __init__(self, model_name):
        self.model_name = model_name
        self.rows = []
        self.y_true = []
        self.y_pred = []

    def add_fold(self, fold, y_train, train_pred, y_test, test_pred):
        row = {
            'train_acc': accuracy_score(y_train, train_pred),
            'acc': accuracy_score(y_test, test_pred),
            'macro_f1': f1_score(y_test, test_pred, average='macro'),
            'benign_recall': recall_score(y_test, test_pred, labels=[0], average='macro'),
        }
        self.rows.append(row)
        self.y_true.extend(y_test)
        self.y_pred.extend(test_pred)
        print(f"Fold {fold}: train acc {row['train_acc']:.4f} | test acc {row['acc']:.4f} | "
              f"macro F1 {row['macro_f1']:.4f} | benign recall {row['benign_recall']:.4f} | test size {len(y_test)}")

    def summary(self, results_dir):
        print(f'\n{self.model_name}, 5-fold cross-validation (mean +- std):')
        for key in ['train_acc', 'acc', 'macro_f1', 'benign_recall']:
            values = [r[key] for r in self.rows]
            print(f'  {key}: {np.mean(values):.4f} +- {np.std(values):.4f}')

        print('\nClassification report (all folds together):')
        print(classification_report(self.y_true, self.y_pred, target_names=CLASS_NAMES, digits=4))

        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        print(conf_matrix)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'{self.model_name} - confusion matrix (5 folds)')
        os.makedirs(results_dir, exist_ok=True)
        filename = 'confusion_matrix_' + self.model_name.lower().replace(' ', '_').replace('+', 'and') + '.png'
        plt.savefig(os.path.join(results_dir, filename), bbox_inches='tight')
        plt.show()
