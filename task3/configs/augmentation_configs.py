from transforms import ElasticDeform, MyColorJitter, MyRandomAffine, MyRandomPerspective
from torchvision.transforms import Compose
from torchvision.transforms import (
    ColorJitter, RandomRotation,
    RandomAffine, RandomPerspective
) 

from utils import make_serializable, make_comp_serializable

# RandomAffine, RandomPerspective, RandomAdjustSharpness, RandomAutocontrast
augmentation_configs = {
    'none': None,
    'ED': ElasticDeform(sigma=1),
    'ED+': ElasticDeform(sigma=5),
    'ED++': ElasticDeform(sigma=10),
    'ED+++': ElasticDeform(sigma=15),
    'ED-': ElasticDeform(sigma=0.1),
    'RandPer': MyRandomPerspective(distortion_scale=0.4),
    'RandAffine': MyRandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(0.1, 0.1, 0.1, 0.1)),
    'RandAffine+': MyRandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(0.2, 0.2, 0.2, 0.2)),
    'ColorJitter': MyColorJitter(brightness=0.02),
    'all': make_comp_serializable(Compose([
            MyRandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(0.1, 0.1, 0.1, 0.1)),
            MyRandomPerspective(distortion_scale=0.4),
            MyColorJitter(brightness=0.02),
            ElasticDeform(sigma=1)
        ])),

    'all_1': make_comp_serializable(Compose([
            MyRandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(0.1, 0.1, 0.1, 0.1)),
            MyRandomPerspective(distortion_scale=0.4),
            ElasticDeform(sigma=1)
        ])),

    'EDAFF': make_comp_serializable(Compose([
            ElasticDeform(sigma=10),
            MyRandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(0.1, 0.1, 0.1, 0.1))
            # MyColorJitter(brightness=0),
        ])),

    'all2': make_comp_serializable(Compose([
            MyRandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(0.1, 0.1, 0.1, 0.1)),
            MyRandomPerspective(distortion_scale=0.4),
            MyColorJitter(brightness=0.02),
            ElasticDeform(sigma=3)
        ])),

    
}