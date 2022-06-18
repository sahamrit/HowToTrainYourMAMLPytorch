"""This script contains experiment configs for MAML++
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
        self.curr_run = None # to be filled during runTime
        self.log_dir = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/HowToTrainYourMAMLPytorch/runs"

        #Dataset Configs
        self.dataset.root_path = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Datasets/MiniImageNet/mini_imagenet_full_size"
        self.dataset.train_path = self.dataset.root_path+"/train"
        self.dataset.test_path = self.dataset.root_path+"/test"
        self.dataset.val_path = self.dataset.root_path+"/val"
        self.dataset.classmap_path = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/HowToTrainYourMAMLPytorch/dataset/classmap.csv"
        self.dataset.sample_train = 0.1
        self.dataset.sample_val= 0.1

        #Model Configs
        self.model.torch_home = "/home/azureuser/cloudfiles/code/Users/asahu.cool/Work/Models"

        #Training Configs
        self.training.inner_loop_steps = 5
        self.training.query_samples_per_class = 2
        self.training.N_way = 5
        self.training.K_shot = 5
        self.training.batch_size = 4
        self.training.image_size = (128,128)
        self.training.gpu = True
        self.training.lr = 0.005
        self.training.inner_lr = 0.05
        self.training.n_epochs = 1
        self.training.train_verbosity = 1
        self.training.val_freq = 2
        self.training.wandb_logging = True




        










