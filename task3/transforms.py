import cv2
import elasticdeform
import torch
import numpy as np
import random
from torchvision.transforms import ColorJitter, RandomAffine, RandomPerspective
import matplotlib.pyplot as plt


def normalize(sample):
    processed = sample - np.amin(sample, (-2,-1))[:, :, np.newaxis, np.newaxis]
    processed = processed / np.amax(processed, (-2,-1))[:, :, np.newaxis, np.newaxis]

    return processed

def equalize(sample):
    processed = cv2.equalizeHist(sample)
    print(processed)

    return processed

# These transforms accept a tensor that has img and label concatenated 
# over the batch dimension
class ElasticDeform():
    def __init__(self, sigma=10, points=3, rotate=0, zoom=0):
        self.sigma = sigma
        self.points = points
        self.rotate = rotate
        self.zoom = zoom

    def __call__(self, sample):
        img, label = sample[0].numpy(), sample[1].numpy()
        # random value in [-self.rotate, self.rotate]
        rotate = 2 * self.rotate * random.random() - self.rotate
        # random value in [1-self.zoom, 1+self.zoom] 
        zoom = 1 + 2 * self.zoom * random.random() - self.zoom

        augmented_img, augmented_label = elasticdeform.deform_random_grid(
            [img, label], 
            points=self.points, sigma=self.sigma, 
            zoom=zoom, rotate=rotate, 
            axis=(1,2) 
        )
        augmented_label = augmented_label.round()

        return torch.stack([torch.tensor(augmented_img), torch.tensor(augmented_label)])

    def serialize(self):
        return {
            'transform_name': 'ElasticDeform',
            'args': {
                'sigma': self.sigma,
                'points': self.points,
                'rotate': self.rotate,
                'zoom': self.zoom
            } 
        }

class MyColorJitter(ColorJitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, sample):
        sample[0] = super().forward(sample[0])
        return sample

    def serialize(self):
        return f"MyColorJitter(brightness={self.brightness})"

class MyRandomAffine(RandomAffine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, sample):
        sample = super().forward(sample)
        return sample

    def serialize(self):
        return f"MyRandomAffine(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear})"

class MyRandomPerspective(RandomPerspective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, sample):
        sample = super().forward(sample)
        return sample

    def serialize(self):
        return f"MyRandomPerspective(distortion_scale={self.distortion_scale}, p={self.p})"