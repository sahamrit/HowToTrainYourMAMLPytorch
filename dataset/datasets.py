"""TODO: This module contains 
"""

import os 
import torchvision
import torch
import random
import math

from pathlib import Path
from typing import *
from PIL import Image
from torch.utils.data import Dataset

from .dataset_utils import *

class MiniImageNetDataset(Dataset):
    """TODO Loads the miniImageNet data in a lazy way where data is read during 
        dataloading via multiple workers and in an episodic way.
    """
    def __init__(self, root_dir: str, N_way: int = 5, K_shot: int = 5,\
        query_samples_per_class: int = 2, transform :Callable = None, sample_frac: float = 1.) -> None:
        """TODO
        """
        self.root_dir = Path(root_dir)
        self.dir = root_dir             
        self.N = N_way
        self.K = K_shot
        self.query_samples_per_class = query_samples_per_class
        self.transform = transform
        
        self.X,self.Y, self.cls_int_map = imagenet_style_dataset_to_XY(self.dir)
        self.episodes = XY_dataset_to_episodes(self.X,\
            self.Y,N_way=N_way,K_shot=K_shot,query_samples_per_class=\
            self.query_samples_per_class)
        
        if sample_frac < 1.:
            self.episodes = random.sample(self.episodes , math.ceil(len(self.episodes)*sample_frac))

    def __len__(self):
        return len(self.episodes)
                   
    
    def __getitem__(self,idx):

        support_set = self.episodes[idx][0]
        query_set = self.episodes[idx][1]           
        
        Xs = support_set[0] ; Ys = support_set[1]
        Xq = query_set[0] ; Yq = query_set[1]

        support_labels = torch.tensor(Ys); query_labels = torch.tensor(Yq)
        support_img_tensor = [] ; query_img_tensor = []

        for img_path in Xs:
            img = torchvision.io.read_image(img_path)
            if self.transform:
                img = self.transform(img.float()/255.0)
            support_img_tensor.append(torch.unsqueeze(img,0))

        for img_path in Xq:
            img = torchvision.io.read_image(img_path)
            if self.transform:
                img = self.transform(img.float()/255.0)
            query_img_tensor.append(torch.unsqueeze(img,0))         

        query_img_tensor = torch.cat(query_img_tensor)
        support_img_tensor = torch.cat(support_img_tensor)

        return ((support_img_tensor,support_labels),(query_img_tensor,query_labels))
