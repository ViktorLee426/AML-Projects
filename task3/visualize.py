import argparse
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from utils import load_zipped_pickle
from dataset import MitralValveDataset
from configs.preprocess_configs import preprocess_configs

from pdb import set_trace

if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-md', '--model_dir', required=True, type=str,
                        help="Path to the model directory.")
    parser.add_argument('-bs', '--batch_size', type=str, default=4, 
                        help="Number of test images to visualize.")
    parser.add_argument('-pc', '--preprocessing_config', required=True, type=str,
                        help="Name of the preprocessing config")
    args = parser.parse_args()

    # Load model
    model = smp.Unet(
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )

    model.load_state_dict(torch.load(f"{args.model_dir}/model.pt"))
    model.eval()

    test_data_path = f"data/test.pkl"
    p_config = preprocess_configs[args.preprocessing_config]
    test_dataset = MitralValveDataset(test_data_path, p_config)

    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    for i, image_batch in enumerate(dataloader):
        fig, axs = plt.subplots(2, args.batch_size, figsize=(20, 20//args.batch_size))

        for i in range(args.batch_size):
            axs[0, i].imshow(image_batch[i].squeeze(), cmap="gray")
            axs[1, i].imshow(image_batch[i].squeeze(), cmap="gray")

        label_batch = model(image_batch).detach() > 0.
        for i in range(args.batch_size):
            axs[1, i].imshow(label_batch[i].squeeze(), cmap="viridis", alpha=0.5)

        plt.show()

        