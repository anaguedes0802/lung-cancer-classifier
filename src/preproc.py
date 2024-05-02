import os
import re
import hashlib
import numpy as np
import cv2
from scipy import ndimage as ndi
from sklearn.model_selection import StratifiedGroupKFold

# number of patients per class, from the dataset description (benign, malignant, normal)
PATIENTS_PER_CLASS = [15, 40, 55]


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # centre crop to a square, some images are 512x623 or 512x801 and
    # stretching them would let the models learn the shape instead of the content
    h, w = image.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return image[top:top + s, left:left + s]


def file_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0


def patient_groups(images, labels, near_duplicate_threshold=3.0):
    # The dataset has no patient ID. The file numbers follow the slice order, so
    # consecutive files are almost always from the same patient. We cut each class
    # at the biggest jumps between consecutive images (one cut less than the number
    # of patients), and then join groups that share near-identical images.
    # small 64x64 copies, made from the 150x150 images used in the first version so the groups stay the same
    small = np.array([cv2.resize(cv2.resize(img, (150, 150)) / 255.0, (64, 64)).flatten() for img in images])
    groups = np.zeros(len(images), dtype=int)
    next_group = 0

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]  # already in file order
        jumps = np.linalg.norm(small[idx[1:]] - small[idx[:-1]], axis=1)
        cuts = set(np.argsort(jumps)[-(PATIENTS_PER_CLASS[c] - 1):])

        for k, i in enumerate(idx):
            if k > 0 and (k - 1) in cuts:
                next_group += 1
            groups[i] = next_group
        next_group += 1

    # union-find to merge groups with near-duplicate images
    parent = list(range(next_group))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(small)):
        dists = np.linalg.norm(small - small[i], axis=1)
        for j in np.where(dists < near_duplicate_threshold)[0]:
            parent[find(groups[i])] = find(groups[j])

    roots = [find(g) for g in groups]
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def load_dataset(dataset_dir):
    # sorted so the labels are always 0 = benign, 1 = malignant, 2 = normal
    classes = sorted(c for c in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, c)))
    images = []
    labels = []
    seen = set()

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)

        for filename in sorted(os.listdir(class_dir), key=file_number):
            image_path = os.path.join(class_dir, filename)

            # the dataset has some exact duplicates, keep only one copy
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in seen:
                continue
            seen.add(file_hash)

            images.append(load_image(image_path))
            labels.append(i)

    labels = np.array(labels)
    groups = patient_groups(images, labels)

    return images, labels, groups


def lung_box(image):
    # rough lung segmentation: the lungs are the two biggest dark regions inside the body
    x = cv2.GaussianBlur(cv2.resize(image, (256, 256)), (5, 5), 0)
    scale = image.shape[0] / 256

    _, body = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = ndi.binary_fill_holes(body > 0)
    lab, n = ndi.label(body)
    if n > 0:
        body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)

    dark = (x < np.percentile(x[body], 35)) & body
    dark = ndi.binary_opening(dark, iterations=2)
    lab, n = ndi.label(dark)
    if n == 0:
        return 0, 0, image.shape[1], image.shape[0]

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes)[-2:]
    mask = np.isin(lab, keep[sizes[keep] > 300])
    if mask.sum() == 0:
        return 0, 0, image.shape[1], image.shape[0]

    ys, xs = np.where(mask)
    pad = 8
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    x1, y1 = min(xs.max() + pad, 255), min(ys.max() + pad, 255)
    return int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)


def dev_test_split(labels, groups):
    # ~20% of the patients are kept aside as the final test set and only used once
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    return next(sgkf.split(np.zeros(len(labels)), labels, groups))


def get_folds(labels, groups, n_splits=5):
    # images from the same (approximate) patient never go to both train and test
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(sgkf.split(np.zeros(len(labels)), labels, groups))
