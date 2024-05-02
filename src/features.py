import numpy as np
import cv2
import torch
import torchvision
import preproc as pr

# ImageNet mean and std, the network was trained with these
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FeatureExtractor:
    # ResNet50 pretrained on ImageNet, used only to turn each image into a vector
    # (it is not trained here). We take features from the whole slice and from
    # the lung region, and join them: 2048 + 2048 = 4096 features per image.

    def __init__(self):
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Identity()  # drop the ImageNet classifier
        self.model.eval()

    def _embed(self, images):
        x = torch.tensor(np.array(images), dtype=torch.float32) / 255.0
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # grayscale -> 3 channels
        x = (x - MEAN) / STD
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 32):
                out.append(self.model(x[i:i + 32]).numpy())
        return np.vstack(out)

    def transform(self, images):
        whole = [cv2.resize(img, (224, 224)) for img in images]

        lungs = []
        for img in images:
            x0, y0, x1, y1 = pr.lung_box(img)
            lungs.append(cv2.resize(img[y0:y1, x0:x1], (224, 224)))

        return np.hstack([self._embed(whole), self._embed(lungs)])
