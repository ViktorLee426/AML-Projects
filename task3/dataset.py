from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from pdb import set_trace

class MitralValveDataset(Dataset):
    def __init__(self, data, device='cpu'):
        """
            data: list of dicts {"image", "label", "video_name", "frame_id"}
        """
        self.device = device
        self.data = data

    def __len__(self):
        return len(self.data)

class MitralValveTrainDataset(MitralValveDataset):
    def __init__(self, data, transform=None, device='cpu'):
        """
            data: list of dicts {"image", "label", "video_name", "frame_id"}
        """
        super(MitralValveTrainDataset, self).__init__(data, device)

        assert "image" in data[0].keys() and "label" in data[0].keys()
        self.transform = transform
        
        for sample in data:
            sample['image'] = torch.tensor(sample["image"], dtype=torch.float32)
            sample['label'] = torch.tensor(sample["label"], dtype=torch.float32)

    def __getitem__(self, idx):
        """
            returns: dict {"image", "label", "video_name", "frame_id"}
        """
        ret_dict = self.data[idx].copy()
        if self.transform:
            prepared_data = torch.cat(
                [ret_dict['image'].clone().detach().unsqueeze(0), 
                ret_dict['label'].clone().detach().unsqueeze(0)],
                axis=0    
            )
            transformed_data = self.transform(prepared_data)

            ret_dict['image'], ret_dict['label'] = \
                transformed_data[0], transformed_data[1].round()

        ret_dict['image'], ret_dict['label'] = \
                ret_dict['image'].to(self.device), ret_dict['label'].to(self.device)
        
        return ret_dict

class MitralValveValDataset(MitralValveDataset):
    def __init__(self, data, device='cpu'):
        """
            data: list of dicts {"image", "label", "video_name", "frame_id"}
        """
        super(MitralValveValDataset, self).__init__(data, device)
        assert "image" in data[0].keys() and "label" in data[0].keys()

        for sample in data:
            sample['image'] = torch.tensor(sample["image"], dtype=torch.float32)
            sample['label'] = torch.tensor(sample["label"], dtype=torch.float32)

    def __getitem__(self, idx):
        """
            returns: dict {"image", "label", "video_name", "frame_id"}
        """
        ret_dict = self.data[idx].copy()
        ret_dict['image'], ret_dict['label'] = \
                ret_dict['image'].to(self.device), ret_dict['label'].to(self.device)
        
        return ret_dict

class MitralValveTestDataset(MitralValveDataset):
    def __init__(self, data, device='cpu'):
        """
            data: list of dicts {"image", "label", "video_name", "frame_id"}
        """
        super(MitralValveTestDataset, self).__init__(data, device)
        assert "image" in data[0].keys() and "label" not in data[0].keys()
        
        for sample in data:
            sample['image'] = torch.tensor(sample["image"], dtype=torch.float32)

    def __getitem__(self, idx):
        """
            returns: dict {"image", "label", "video_name", "frame_id"}
        """
        ret_dict = self.data[idx].copy()
        ret_dict['image'] = ret_dict['image'].to(self.device)
        
        return ret_dict