"""This script contains implementation of MiniImageNetDataset

It contains the following class:
    *MiniImageNetDataset: Class inheriting from torch.utils.data.Dataset
        overloads implementation of __getitem__ and __len__
    
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
    """Loads the miniImageNet data in a lazy way, where data is read during 
        dataloading. Each call of __getitem__(self,idx) returns one episode/
        task which contains the support and query set images, labels for that
        episode

        Attributes
        ----------
        dir: pathlib.Path
            root_dir in ImageNet Directory Structure(refer below)
        N: int
            N in N way K shot learning
        K: int
            K in N way K shot learning
        query_samples_per_class: int
            No of query samples per class in query set
        transform: Callable
            transform = torch.nn.Sequential(
                transforms.Resize(config.training.image_size),
                transforms.Normalize(ImageNetMean,ImageNetVariance),
                ...
            )
        cls_int_map: Dict[str,int]
            Integer map of string classes
        episodes: List[Tuple[Tuple[List[str],List[int]],Tuple[List[str],List[int]]]]
            list of episodes, where each episode consists of tuple of support set 
            and query set. Each set consists of list of image paths and their labels  

        Methods
        ------- 
        __len__(self):
            Returns total number of episodes of mini ImageNet dataset
        __getitem__(self,idx):
            Returns Tuple[Tuple[torch.Tensor,torch.Tensor],Tuple[torch.Tensor,torch.Tensor]]
            ((xs,ys),(xq,yq)) where xs/xq denote support and query set image tensors
            and ys/yq denotes support and query set labels
            Shapes:
                xs -> [N_way*K_shot,C,H,W]
                ys -> [N_way*K_shot]
                xq -> [N_way*query_samples_per_class,C,H,W]
                yq -> [N_way*query_samples_per_class]
        Additonal Information
        ---------------------
        ImageNet Directory Structure:
            root_dir
            |--train
            |  |--cls1
            |  |  |--a.png
            |  |--cls2
            |  |  |--x.png
            |--test
            |  |--cls3
            |  |  |--b.png
            |  |--cls4
            |  |  |--y.png
            |--val
            |  |--cls5
            |  |  |--c.png
            |  |--cls6
            |  |  |--z.png  
    """
    def __init__(self, root_dir: str, N_way: int = 5, K_shot: int = 5,\
        query_samples_per_class: int = 2, transform :Callable = None, sample_frac: float = 1.) -> None:
        """
        Parameters
        ----------
            root_dir: pathlib.Path
                root_dir in ImageNet Directory Structure(refer below)
            N_way: int
                N in N way K shot learning
            K_shot: int
                K in N way K shot learning
            query_samples_per_class: int
                No of query samples per class in query set
            transform: Callable
                transform = torch.nn.Sequential(
                    transforms.Resize(config.training.image_size),
                    transforms.Normalize(ImageNetMean,ImageNetVariance),
                    ...
                )
            sample_frac: float
                fraction of episodes to use for this dataset
        """
        
        self.dir = Path(root_dir)             
        self.N = N_way
        self.K = K_shot
        self.query_samples_per_class = query_samples_per_class
        self.transform = transform
        
        X, Y, self.cls_int_map = imagenet_style_dataset_to_XY(self.dir)
        self.episodes = XY_dataset_to_episodes(X,\
            Y, N_way=N_way, K_shot=K_shot, query_samples_per_class=\
            self.query_samples_per_class)
        
        if sample_frac < 1.:
            self.episodes = random.sample(self.episodes , math.ceil(len(self.episodes)*sample_frac))

    def __len__(self):
        """Returns total number of episodes of mini ImageNet dataset
        """

        return len(self.episodes)
                   
    
    def __getitem__(self,idx):
        """Returns Tuple[Tuple[torch.Tensor,torch.Tensor],Tuple[torch.Tensor,torch.Tensor]]
            ((xs,ys),(xq,yq)) where xs/xq denote support and query set image tensors
            and ys/yq denotes support and query set labels
            Shapes:
                xs -> [N_way*K_shot,C,H,W]
                ys -> [N_way*K_shot]
                xq -> [N_way*query_samples_per_class,C,H,W]
                yq -> [N_way*query_samples_per_class]
        """

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
