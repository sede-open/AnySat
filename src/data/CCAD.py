from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pickle
import json

def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in ["s2"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output


class CCAD(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path=None,
        ignore_index=-1,
        temporal_dropout=0,
    ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (callable): transform to apply to the data
            split (str): split to use (train, val, test)
            norm_path (str): path to the normalization file
            temporal_dropout (float): probability of dropping a temporal sample
        """

        # check the split and assign the path
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of ['train', 'val', 'test']")
        if split == "train":
            path_split = os.path.join(path, "train_fold.csv")
        elif split == "val":
            path_split = os.path.join(path, "val_fold.csv")
        elif split == "test":
            path_split = os.path.join(path, "test_fold.csv")
        
        self.path = path
        self.path_split = path_split
        self.modalities = modalities
        self.transform = transform
        self.split = split
        self.norm_path = norm_path
        self.temporal_dropout = temporal_dropout
        self.path_df = pd.read_csv(self.path_split, header=None)
        self.ignore_index = ignore_index
        self.collate_fn = collate_fn
        self.norm = None
        if norm_path is not None:
            norm = {}
            for modality in self.modalities:
                file_path = os.path.join(norm_path, "NORM_{}_patch.json".format(modality))
                if not(os.path.exists(file_path)):
                    self.compute_norm_vals(norm_path, modality)
                normvals = json.load(open(file_path))
                norm[modality] = (
                    torch.tensor(normvals['mean']).float(),
                    torch.tensor(normvals['std']).float(),
                )
            self.norm = norm
    def load_data(self, path):
        """
        Loads the data from the given path.
        Args:
            path (str): path to the data
        Returns:
            list: list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
        """
        path = os.path.join(self.path, path)
        with open(path, 'rb') as file:
            data = pickle.load(file)
        file_name = path.split('/')[-1].split('.')[0]

        return data, file_name
    
    def __getitem__(self, index):
        """
        Returns the item at the given index.
        Args:
            index (int): index of the item to return
        Returns:
            dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
        """
        # read the csv file
        path_i = self.path_df.iloc[index].values[0]

        row, name_file = self.load_data(path_i)

        # get the label and name
        label = row["labels"][0]
        img = torch.Tensor(row["img"])
        days = torch.Tensor(row["doy"])
        name = name_file
        
        if self.split == "train" and self.temporal_dropout > 0:
            N = len(days)
            
            random_indices = torch.randperm(N)[:(int(N * (1 - self.temporal_dropout)))]
            days = days[random_indices]
            img = img[random_indices]

        output = {
            "s2": img,
            "s2_dates": torch.Tensor(days),
            "label": torch.Tensor(label).long(),
            'name': name, 
        }
            
        
        if self.norm is not None:
            for modality in self.modalities:
                if len(output[modality].shape) == 4:
                    output[modality] = (output[modality] - self.norm[modality][0][None, :, 
                                                None, None]) / self.norm[modality][1][None, :, None, None]
                else:
                    output[modality] = (output[modality] - self.norm[modality][0][:, None, None]) / self.norm[modality][1][:, None, None]
        
        return self.transform(output)
    
    def __len__(self):
        return len(self.path_df)