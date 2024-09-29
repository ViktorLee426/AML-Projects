import random
import time
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import json

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import MitralValveTrainDataset, MitralValveValDataset
from configs.preprocess_configs import preprocess_configs
from configs.augmentation_configs import augmentation_configs
from configs.model_configs import model_configs
from utils import iou, logits_to_binary
import time

from torch.utils.tensorboard import SummaryWriter

from pdb import set_trace

# Reproducibility
seed = 3141592
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

DATA_PATH = "data"

#TODO add custom model (maybe)
#TODO cross validation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-pc', '--pre_config', required=True, type=str,
                        help="Name of the preprocessing config in configs/preprocess_configs "\
                            "used for this experiment.")
    parser.add_argument('-ac', '--aug_config', required=True, type=str,
                        help="Name of the data augmentation config in configs/augmentation_configs "\
                            "used for this experiment.")
    parser.add_argument('-mc', '--model_config', required=True, type=str,
                        help="Name of the model config in configs/model_configs "\
                            "used for this experiment.")
    parser.add_argument('-no_save', '--no_save', action="store_true",
                        help="Do not save experiment.")
    args = parser.parse_args()

    experiment_name = f"{args.pre_config}-{args.aug_config}-{args.model_config}"

    p_config = preprocess_configs[args.pre_config]
    m_config = model_configs[args.model_config]
    augmentation = augmentation_configs[args.aug_config]

    # Load and preprocess data
    train_data_path = f"{DATA_PATH}/train.pkl"
    preprocessor = Preprocessor(train_data_path, p_config)
    training_samples, val_samples = preprocessor.create_dataset(5, 0)
    train_dataset = MitralValveTrainDataset(training_samples, transform=augmentation, device=device)
    val_dataset = MitralValveValDataset(val_samples, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=m_config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=m_config['batch_size'], shuffle=True)

    # Prepare the model
    model = m_config['model'](**m_config['model_hyperparams'])
    model.to(device)

    # loss_func = m_config['loss'](**m_config['loss_hyperparams'], reduction='sum')
    loss_func = m_config['loss'](**m_config['loss_hyperparams'])
    optimizer = m_config['optimizer'](model.parameters(), **m_config['optimizer_hyperparams'])

    # Create experiment folder and save results
    if not args.no_save:
        if os.path.isdir(f"results/{experiment_name}"):
            attempt_num = 1
            while os.path.isdir(f"results/{experiment_name}-{attempt_num}"):
                attempt_num += 1
            experiment_name = f"{experiment_name}-{attempt_num}"

        print(f"[INFO] Experiment name is {experiment_name}")
        os.mkdir(f"results/{experiment_name}")

        prep_path = f"results/{experiment_name}/p_config.json"
        with open(prep_path, 'w') as f:
            json.dump(p_config, f, default=lambda o: o.serialize())

        aug_path = f"results/{experiment_name}/aug_config.json"
        with open(aug_path, 'w') as f:
            json.dump(augmentation, f, default=lambda o: o.serialize())

        model_path = f"results/{experiment_name}/m_config.json"
        with open(model_path, 'w') as f:
            json.dump(m_config, f, default=lambda o: str(o))

    # === TRAINING ===
    print("[INFO] Training begins")

    best_iou = -1.

    start_time = time.time()
    if not args.no_save:
        writer = SummaryWriter(log_dir=f"results/{experiment_name}")
    for epoch in tqdm(range(m_config['epochs']), disable=True):
        epoch_start_time = time.time()

        train_losses = []
        train_ious = []
        val_losses = []
        val_ious = []
        # Train
        model.train()
        for (i, data) in enumerate(train_dataloader):
            image_batch, label_batch = data['image'], data['label']
            pred = model(image_batch)
            loss = loss_func(pred, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.cpu().detach().numpy())
            discrete_pred = logits_to_binary(pred)
            train_ious += iou(discrete_pred.cpu().detach().numpy(), label_batch.cpu().detach().numpy())

        if not args.no_save:
            if 'window' in p_config.keys() and p_config['window'] > 1:
                pass
            else:
                writer.add_image("images/train_pred", discrete_pred[0], epoch)
                writer.add_image("images/train_label", label_batch[0], epoch)

        # Eval
        with torch.no_grad():
            model.eval()

            for data in val_dataloader:
                image_batch, label_batch = data['image'], data['label']
                pred = model(image_batch)
                discrete_pred = logits_to_binary(pred)
                discrete_pred = discrete_pred.cpu().detach().numpy()

                if 'post_kernel' in m_config.keys() and ('window' not in m_config.keys()):
                    for i in range(discrete_pred.shape[0]):
                        discrete_pred[i] = cv2.morphologyEx(discrete_pred[i], cv2.MORPH_CLOSE, m_config['post_kernel'])

                val_ious += iou(discrete_pred, label_batch.cpu().detach().numpy(), p_config=p_config)
                val_losses.append(loss_func(pred, label_batch).cpu().detach().numpy())
        
        avg_train_loss = np.mean(train_losses)
        avg_train_iou = np.mean(train_ious)
        avg_val_loss = np.mean(val_losses)
        avg_val_iou = np.mean(val_ious)

        print("[INFO] EPOCH: {}/{}".format(epoch + 1, m_config['epochs']))
        print(f"Total runtime {time.time() - epoch_start_time}")
        print("Train loss: {:.6f}, train iou: {:.4f}, validation loss: {:.4f}, val iou: {:.4f}"\
            .format(avg_train_loss, avg_train_iou, avg_val_loss, avg_val_iou))
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            if not args.no_save:
                torch.save(model.state_dict(), f"results/{experiment_name}/model.pt")

        if not args.no_save:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('IOU/train', avg_train_iou, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('IOU/val',avg_val_iou, epoch)
            writer.add_scalar('IOU/best_val',best_iou, epoch)
            
            if 'window' in p_config.keys() and p_config['window'] > 1:
                pass
            else:
                writer.add_image("images/discrete_pred", discrete_pred[0], epoch)
                writer.add_image("images/label", label_batch[0], epoch)


    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

    
