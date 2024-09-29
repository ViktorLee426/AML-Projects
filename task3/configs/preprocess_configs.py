from torchvision.transforms import Compose
from transforms import equalize
from utils import make_serializable, make_comp_serializable

# img_size must be divisible by 32 for unet
preprocess_configs = {
    'simple': {
        'img_size': 128,
        'use_amateur': True,
        'transform': make_comp_serializable(Compose([]))
    },

    '128_win5': {
        'img_size': 128,
        'use_amateur': True,
        'window': 5,
        'pred_path': 'results/simple-none-smp_unet_jacc/pred_train.pkl',
        'val_expert_only': False,
        'human_val': True,
    },

    '128_E': {
        'img_size': 128,
        'use_amateur': False,
    },

    '128_A': {
        'img_size': 128,
        'use_amateur': True,
    },

    '128_AE': {
        'img_size': 128,
        'use_amateur': True,
        'val_expert_only': True
    },

    '128_AE_added': {
        'img_size': 128,
        'use_amateur': True,
        'val_expert_only': True,
        'pred_path': 'results/128_AE-all-unet++/pred_train.pkl',
    },

    '128_AE_added_1': {
        'img_size': 128,
        'use_amateur': True,
        'val_expert_only': True,
        'pred_path': 'results/128_AE-all-unet++/pred_train.pkl',
        'val_orig_labels': True
    },

    '128_EE_added': {
        'img_size': 128,
        'use_amateur': False,
        'val_expert_only': True,
        'pred_path': 'results/128_AE-all-unet++/pred_train.pkl',
        'val_orig_labels': True
    },

    '128_AE_added_win5': {
        'img_size': 128,
        'use_amateur': True,
        'val_expert_only': True,
        'pred_path': 'results/128_AE-all-unet++/pred_train.pkl',
        'window': 5,
        'human_val': True,
    },

    '256_E': {
        'img_size': 256,
        'use_amateur': False,
    },

    '256_A': {
        'img_size': 256,
        'use_amateur': True,
    },

    '128_A_add_labels': {
        'img_size': 128,
        'use_amateur': True,
        'pred_path': 'results/'
    },
    
    # Equalization doesn't seem to be needed
    'advanced_eq': {
        'img_size': 256,
        'use_amateur': False,
        'transform': make_comp_serializable(Compose([
            make_serializable(equalize, 'equalize'),
        ]))
    }
}