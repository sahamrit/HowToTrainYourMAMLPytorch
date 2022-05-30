"""TODO
"""

import os
import json

from pathlib import Path
from types import SimpleNamespace

class ExperimentConfig(SimpleNamespace):
    def __init__(self) -> None:
        
        self.dataset = SimpleNamespace()
        self.model = SimpleNamespace()
        self.training = SimpleNamespace()


        #General Configs 
        self.seed = 0

        #Dataset Configs
        self.dataset.root_path = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Datasets/MiniImageNet/mini_imagenet_full_size"
        self.dataset.train_path = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Datasets/MiniImageNet/mini_imagenet_full_size/train"
        self.dataset.test_path = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Datasets/MiniImageNet/mini_imagenet_full_size/test"
        self.dataset.val_path = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Datasets/MiniImageNet/mini_imagenet_full_size/val"
        self.dataset.classmap_path = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Datasets/MiniImageNet/classmap.csv"

        #Model Configs
        self.model.torch_home = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Models"

        #Training Configs
        self.training.inner_loop_steps = 4
        self.training.query_samples_per_class = 2
        self.training.N_way = 5
        self.training.K_shot = 5
        self.training.batch_size = 16
        self.training.image_size = (128,128)
        self.training.gpu = True
        self.training.lr = 0.01
        self.training.n_epochs = 5




        










