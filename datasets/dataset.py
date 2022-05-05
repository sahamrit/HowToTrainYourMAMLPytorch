"""TODO: This module contains 
"""

import os 
import torchvision
from pathlib import Path
from typing import *
from utils import *


class MiniImageNetDataset:
    """TODO
    """
    def __init__(self,root_dir: str,N_way: int = 5, K_shot: int = 1) -> None:
        """TODO
        """
        self.root_dir = Path(root_dir)
        self.train_dir = os.path.join(root_dir,"train")
        self.test_dir = os.path.join(root_dir,"test")
        self.val_dir = os.path.join(root_dir,"val")
        self.N = N_way
        self.K = K_shot
        self.train_xy = imagenet_style_dataset_to_XY(self.train_dir)
        self.val_xy = imagenet_style_dataset_to_XY(self.val_dir)
        self.test_xy = imagenet_style_dataset_to_XY(self.test_dir)
    
    def __len__(self):
        pass
    
    def __getitem__(self,idx):
        pass

