import cv2
import torch.nn as nn
from torch.optim import Adam
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss

# Losses: 
model_configs = {
    #OLD don't use
    'smp_unet': {
        'model': smp.Unet,
        'model_hyperparams': {
            'encoder_weights': None,
            'in_channels': 1,
            'classes': 1,
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-3
        },

        'loss': nn.BCEWithLogitsLoss,
        'loss_hyperparams': {},

        'batch_size': 8,
        'epochs': 100,
    },

    'smp_unet_jacc': {
        'model': smp.Unet,
        'model_hyperparams': {
            'encoder_weights': None,
            'in_channels': 1,
            'classes': 1,
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-3
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    'smp_unet_reg': {
        'model': smp.Unet,
        'model_hyperparams': {
            'encoder_weights': None,
            'in_channels': 1,
            'classes': 1,
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

        
    'unet++': {
        'model': smp.UnetPlusPlus,
        'model_hyperparams': {
            'encoder_weights': None,
            'in_channels': 1,
            'classes': 1,
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },
    
    'unet++_imagenet': {
        'model': smp.UnetPlusPlus,
        'model_hyperparams': {
            'encoder_weights': 'imagenet',
            'in_channels': 1,
            'classes': 1,
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },
        'batch_size': 8,
        'epochs': 100,
    },

    'unet++_resnet50': {
        'model': smp.UnetPlusPlus,
        'model_hyperparams': {
            'encoder_name': 'resnet50',
            'encoder_weights': None,
            'in_channels': 1,
            'classes': 1,
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    'smp_unet_jacc_c5': {
        'model': smp.Unet,
        'model_hyperparams': {
            'encoder_weights': None,
            'in_channels': 5,
            'classes': 5,
        },
        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-3,
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    
    'unet++_attention': {
        'model': smp.UnetPlusPlus,
        'model_hyperparams': {
            'encoder_weights': None,
            'in_channels': 1,
            'classes': 1,
            'decoder_attention_type': 'scse',
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    'unet++_512': {
        'model': smp.UnetPlusPlus,
        'model_hyperparams': {
            'encoder_weights': None,
            'in_channels': 1,
            'classes': 1,
            'decoder_channels': (512, 256, 128, 64, 32)
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    'manet': {
        'model': smp.MAnet,
        'model_hyperparams': {
            'encoder_weights': None,
            'encoder_depth': 5,
            'in_channels': 1,
            'classes': 1,
            'decoder_channels': (512, 256, 128, 64, 32)
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    'manet_small': {
        'model': smp.MAnet,
        'model_hyperparams': {
            'encoder_weights': None,
            'encoder_depth': 3,
            'in_channels': 1,
            'classes': 1,
            'decoder_channels': (128, 64, 32)
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    'linknet': {
        'model': smp.Linknet,
        'model_hyperparams': {
            'encoder_weights': None,
            'encoder_depth': 5,
            'in_channels': 1,
            'classes': 1,
            # 'decoder_channels': (512, 256, 128, 64, 32)
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 100,
    },

    'fpn': {
        'model': smp.FPN,
        'model_hyperparams': {
            'encoder_weights': None,
            'encoder_depth': 5,
            'in_channels': 1,
            'classes': 1,
            # 'decoder_channels': (512, 256, 128, 64, 32)
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 50,
    },

    'dlv3': {
        'model': smp.DeepLabV3,
        'model_hyperparams': {
            'encoder_weights': None,
            'encoder_depth': 5,
            'in_channels': 1,
            'classes': 1,
            'decoder_channels': (512, 256, 128, 64, 32)
        },

        'optimizer': Adam,
        'optimizer_hyperparams': {
            'lr': 1e-4,
            'weight_decay': 1e-5
        },

        'loss': JaccardLoss,
        'loss_hyperparams': {
            'mode': 'binary'
        },

        'batch_size': 8,
        'epochs': 50,
    },
    

# 'unet++_close': {
#         'model': smp.UnetPlusPlus,
#         'model_hyperparams': {
#             'encoder_weights': None,
#             'in_channels': 1,
#             'classes': 1,

#         'post_kernel': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
#     },
}