import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pickle
import gzip
from pdb import set_trace

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def get_box_bounds(box):
    u = min(i for i in range(box.shape[0]) if np.any(box[i]))
    d = max(i for i in range(box.shape[0]) if np.any(box[i]))
    l = min(i for i in range(box.shape[0]) if np.any(box[:,i]))
    r = max(i for i in range(box.shape[0]) if np.any(box[:,i]))
    return l, r, u, d
    
def add_box_to_frame(frame, l, r, u, d):
    frame[u-1:u+2, l:r+1] = [255, 0, 0]
    frame[d-1:d+2, l:r+1] = [255, 0, 0]
    frame[u:d+1, l-1:l+2] = [255, 0, 0]
    frame[u:d+1, r-1:r+2] = [255, 0, 0]

def display_datapoint(data):
    print(data['dataset'])
    video = data['video']
    label = data['label']
    box = data['box']
    imageio.mimwrite('test.mp4', video, fps=30); 
    
    for i, frame in enumerate(data['frames']):
        plt.subplot(1, 3, i+1)
        rgb = cv2.cvtColor(video[frame], cv2.COLOR_GRAY2RGB)
        rgb[label[frame]] = [0, 255, 0]
        l, r, u, d = get_box_bounds(box)
        add_box_to_frame(rgb, l, r, u, d)
        plt.imshow(rgb)

def iou(pred_batch, label_batch, reduction='none', p_config=None):
    num_batches = pred_batch.shape[0]
    assert num_batches == label_batch.shape[0]
    
    scores = np.zeros(num_batches)
    for i in range(num_batches):
        pred = pred_batch[i, :]
        label = label_batch[i, :]
        if p_config is not None and 'human_val' in p_config and p_config['human_val']:
            pred = pred[0]
            label = label[0]
        intersection = (pred * label).sum()
        union = ((pred + label) > 0).sum()
        if union > 0:
            scores[i] = intersection / union
        else:
            scores[i] = 0

    if reduction == 'none':
        return scores.tolist()
    elif reduction == 'mean':
        return np.mean(scores)
    else:
        raise ValueError(f"Unsuported type of reduction: {reduction}. Must be either 'none' or 'sum'")

def logits_to_binary(x):
    return (x > 0).float()

def to_input_shape(data):
    # H * W * T to T * 1 * H * W
    return np.expand_dims(np.transpose(data, axes=(2,0,1)), axis=1)

def to_data_shape(output):
    # convert tensor of shape T * 1 * H * W to tensor of shape H * W * T
    return output.squeeze().permute(1,2,0)

def resize(imgs, target_height, target_width, interpolation, dtype=float):
    output = cv2.resize(imgs * 1., (target_width, target_height), interpolation = interpolation)
    return np.array(output, dtype=dtype)

def make_serializable(obj, str):
    obj.serialize = lambda: str
    return obj

def make_comp_serializable(comp):
    comp.serialize = lambda: {
        'transform_name': "Comp",
        'childred': [t.serialize() for t in comp.transforms]
    }
    return comp

def get_video_sizes(data):
    """
        return: a dictionary with video name as key and (H, W, T) as value
    """
    video_sizes = {}
    for d in data:
        video_sizes[d['name']] = d['video'].shape
    return video_sizes