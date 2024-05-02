import os
import time
import numpy as np
import torch
import preproc as pr
from features import FeatureExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

torch.manual_seed(42)

start = time.time()
images, labels, groups = pr.load_dataset(os.path.join(ROOT_DIR, 'data'))
print(f'{len(images)} images after removing duplicates, per class (benign, malignant, normal): {np.bincount(labels)}')
print(f'{len(np.unique(groups))} approximate patient groups')

X = FeatureExtractor().transform(images)
print('Feature matrix:', X.shape)

os.makedirs(os.path.join(ROOT_DIR, 'features'), exist_ok=True)
np.savez(os.path.join(ROOT_DIR, 'features', 'features.npz'), X=X, labels=labels, groups=groups)
print(f'Done in {time.time() - start:.0f} s')
