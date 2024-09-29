import cv2
import numpy as np
from matplotlib import pyplot as plt
from transforms import normalize
from pdb import set_trace

from utils import load_zipped_pickle, get_video_sizes

class Preprocessor():
    def __init__(self, data, config, training=None):
        self.config = config
        self.preprocessed_data = []
        self.prediction_path = None if 'pred_path' not in self.config.keys() else self.config['pred_path']
        self.window = 1 if 'window' not in self.config.keys() else self.config['window']
        if self.window > 1:
            assert self.prediction_path is not None # only use window when prediction_path is available
        self.human_val = False if 'human_val' not in self.config.keys() else self.config['human_val']

        data_path = data
        print(f"[INFO] loading data from {data_path}")
        unprocessed_data = load_zipped_pickle(data_path)
        if self.prediction_path is not None:
            predictions = load_zipped_pickle(self.prediction_path)

        self.original_video_sizes = get_video_sizes(unprocessed_data)
        self.training = ('label' in unprocessed_data[0].keys()) if training is None else training

        for i, video_dict in enumerate(unprocessed_data):
            if self.training and video_dict['dataset'] == 'amateur' and not self.config['use_amateur']:
                continue
            
            video_dict['video'] = self._preprocess_video(video_dict['video'])

            # Assert that all labels and bboxes contain only ones and zeros
            if self.training:
                video_dict['label'] = self._preprocess_label(video_dict['label'])
                video_dict['box'] = self._preprocess_box(video_dict['box'])
                assert np.sum(np.logical_or(video_dict['label'] == 0, video_dict['label'] == 1)) == video_dict['label'].size
                assert np.sum(np.logical_or(video_dict['box'] == 0, video_dict['box'] == 1)) == video_dict['box'].size
                if self.prediction_path is not None:
                    predicted_labels = self._preprocess_label(predictions[i]['prediction'])
                    predicted_labels[video_dict['frames'],:,:,:] = video_dict['label'][video_dict['frames'],:,:,:]
                    video_dict['label'] = predicted_labels

            self.preprocessed_data.append(video_dict)
    
    #TODO combine multiple frames here here
    def create_dataset(self, folds = 1, fold_id = 0):
        if self.training:
            train_samples = []
            val_samples = []

            if self.config['val_expert_only']:
                expert_ids = []
                for i, video_dict in enumerate(self.preprocessed_data):
                    if video_dict['dataset'] == 'expert':
                        expert_ids.append(i)
                val_size = len(expert_ids) // folds if folds > 1 else 0
                val_ids = expert_ids[val_size * fold_id: val_size * (fold_id + 1)]
            else:
                val_size = len(self.preprocessed_data) // folds if folds > 1 else 0
                val_ids = np.arange(val_size * fold_id, val_size * (fold_id + 1))
            
            for i, video_dict in enumerate(self.preprocessed_data):
                if self.prediction_path is not None:
                    if 'val_orig_labels' in self.config.keys() and self.config['val_orig_labels'] and i in val_ids:
                        used_frames = video_dict['frames']
                    else:
                        used_frames = range(video_dict['video'].shape[0] - self.window + 1)
                else:
                    used_frames = video_dict['frames']
                
                for frame in used_frames:
                    frames = list(range(frame, frame + self.window)) # frames = [frame, ..., frame+window-1]
                    sample = {
                        'image': video_dict['video'][frames, :, :, :].squeeze(axis=1),
                        'label': video_dict['label'][frames, :, :, :].squeeze(axis=1),
                        'video_name': video_dict['name'],
                        'frame_id': frame,
                        'expert': video_dict['dataset'] == 'expert'
                    }
                    if self.human_val:
                        if frame in video_dict['frames']:
                            sample['dataset_type'] = 'val'
                            val_samples.append(sample)
                        else:
                            sample['dataset_type'] = 'train'
                            train_samples.append(sample)
                    else:
                        if i in val_ids:
                            sample['dataset_type'] = 'val'
                            val_samples.append(sample)
                        else:
                            sample['dataset_type'] = 'train'
                            train_samples.append(sample)
            return train_samples, val_samples

        else:
            test_samples = []
            for i, video_dict in enumerate(self.preprocessed_data):
                for frame in range(video_dict['video'].shape[0] - self.window + 1):
                    frames = list(range(frame, frame + self.window))
                    sample = {
                        'image': video_dict['video'][frames, :, :, :].squeeze(axis=1),
                        'video_name': video_dict['name'],
                        'frame_id': frame,
                        'expert': True if 'dataset' not in video_dict else video_dict['dataset'] == 'expert',
                        'dataset_type': 'test',
                    }
                    test_samples.append(sample)
            return test_samples
            
    def _preprocess_video(self, x):
        # H * W * T to T * 1 * H * W
        x_resized = cv2.resize(x * 1., (self.config['img_size'], self.config['img_size']), interpolation = cv2.INTER_LINEAR)
        x_reshaped = np.expand_dims(np.transpose(x_resized, axes=(2,0,1)), axis=1)
        x_normalized = normalize(x_reshaped)

        if 'transform' in self.config.keys():
            x_transformed = self.config['transform'](x_normalized)
            return x_transformed
        else:
            return x_normalized

    def _preprocess_label(self, x):
        # H * W * T to T * 1 * H * W
        x_resized = cv2.resize(x * 1., (self.config['img_size'], self.config['img_size']), interpolation = cv2.INTER_NEAREST)
        x_reshaped = np.expand_dims(np.transpose(x_resized, axes=(2,0,1)), axis=1)
        
        return x_reshaped

    def _preprocess_box(self, x):
        # H * W to H * W
        x_resized = cv2.resize(x * 1., (self.config['img_size'], self.config['img_size']), interpolation = cv2.INTER_NEAREST)

        return x_resized