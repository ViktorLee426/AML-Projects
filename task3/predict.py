import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from pdb import set_trace

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from utils import load_zipped_pickle, save_zipped_pickle, to_input_shape, to_data_shape, resize, get_video_sizes
from dataset import MitralValveDataset, MitralValveTrainDataset, MitralValveTestDataset
from configs.preprocess_configs import preprocess_configs
from configs.model_configs import model_configs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--experiment_name', required=True, type=str,
                        help="Name of the experiment on which to perform inference.")
    parser.add_argument('-bs', '--batch_size', type=int, default=16, 
                        help="Number of test images to process in batch.")
    parser.add_argument('-train', '--predict_train', action='store_true',
                        help="Use the model to label training data")
    parser.add_argument('-no_save', '--no_save', action="store_true",
                        help="Do not save experiment.")
    args = parser.parse_args()

    p_config_name = args.experiment_name.split('-')[0]
    m_config_name = args.experiment_name.split('-')[2]
    p_config = preprocess_configs[p_config_name]
    m_config = model_configs[m_config_name]
    
    # Load model
    model = m_config['model'](**m_config['model_hyperparams'])
    model.to(device)
    model.load_state_dict(torch.load(f"results/{args.experiment_name}/model.pt", map_location=device))
    model.eval()

    if args.predict_train:
        data_path = "data/train.pkl"
    else:
        data_path = "data/test.pkl"

    preprocessor = Preprocessor(data_path, p_config, training=False)
    video_sizes = preprocessor.original_video_sizes
    samples = preprocessor.create_dataset(1)

    dataset = MitralValveTestDataset(samples, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    buffer = defaultdict(lambda: defaultdict(list)) # {VIDEO_NAME: {FRAME_ID: [1*H*W TENSOR, ...], ...}}

    def write_buffer(buffer, video_name_batch, frame_id_batch, prediction_batch):
        batch_size, n_channels = prediction_batch.shape[:2]
        for i, (video_name, frame_id) in enumerate(zip(video_name_batch, frame_id_batch)):
            prediction = prediction_batch[i] # C * H * W
            for channel in range(n_channels):
                pred = prediction[channel].unsqueeze(0) # 1 * H * W
                buffer[video_name][frame_id + channel].append(pred)
    
    def to_predictions(buffer):
        predictions = {} # {VIDEO_NAME: [C*H*W TENSOR, ...]}
        for video_name in buffer.keys():
            for frame_id in range(len(buffer[video_name])):
                assert frame_id in buffer[video_name]
                video_frame_pred_list = buffer[video_name][frame_id] # N * 1 * H * W
                maj_vote, _ = torch.stack(video_frame_pred_list).mode(dim=0) # 1 * H * W
                if video_name not in predictions:
                    predictions[video_name] = []
                predictions[video_name].append(maj_vote)
        return predictions

    for i, batch in enumerate(tqdm(dataloader)):
        image_batch = batch['image'] # B * C * H * W
        video_name_batch = batch['video_name']
        frame_id_batch = batch['frame_id']
        prediction_batch = model(image_batch).detach() > 0.

        write_buffer(buffer, video_name_batch, frame_id_batch.tolist(), prediction_batch)
    
    predictions = to_predictions(buffer)
    
    output = []
    for video_name in predictions:
        target_height, target_width, video_length = video_sizes[video_name]
        pred_stacked = torch.stack(predictions[video_name])
        pred_stacked_reshaped = to_data_shape(pred_stacked)
        pred_stacked_reshaped_resized = resize(pred_stacked_reshaped.cpu().detach().numpy(), target_height, target_width, cv2.INTER_NEAREST, dtype=bool)
        assert pred_stacked_reshaped_resized.shape[-1] == video_length
        output.append({
            'name': video_name,
            'prediction': pred_stacked_reshaped_resized
        })

    if not args.no_save:
        save_zipped_pickle(output, f'results/{args.experiment_name}/pred{"_train" if args.predict_train else "_test"}.pkl')








