#Open source specific imports
import os 
import json
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
import wandb
import datetime

from pathlib import Path
from typing import *
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from types import SimpleNamespace

#Local code specific imports
from utils import *
from configs import *
from dataset.datasets import MiniImageNetDataset
from engine.trainer import do_train
from engine.optimizers import AdamExplicitGrad

def main():
    config = ExperimentConfig()
    if config.curr_run is None:
        config.curr_run = str(datetime.datetime.now())

    if config.training.wandb_logging:
        wandb.init(
            project="maml++",
            name=f"experiment_{config.curr_run}",
            config=config)

    #Setting seeds for reproducability
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    #Global Variables
    os.environ['TORCH_HOME'] = config.model.torch_home

    #Imagenet mean and var to normalise the input images
    ImageNetMean, ImageNetVariance = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    #Human readable class mapping of ImageNet classes
    class_map = pd.read_csv(config.dataset.classmap_path, header = None, delimiter=" ")
    class_map = dict(zip(class_map[0],class_map[1]))

    print(config)

    transform = torch.nn.Sequential(
        transforms.Resize(config.training.image_size),
        transforms.Normalize(ImageNetMean,ImageNetVariance)
    )

    # TODO Dataset Reproducability incase training interupts
    mini_imagenet_dataset_train = MiniImageNetDataset(config.dataset.train_path,\
        N_way=config.training.N_way,\
            K_shot=config.training.K_shot,\
                query_samples_per_class=config.training.query_samples_per_class, \
                    transform= transform, sample_frac = 0.4)

    mini_imagenet_dataset_val = MiniImageNetDataset(config.dataset.val_path,\
        N_way=config.training.N_way,\
            K_shot=config.training.K_shot,\
                query_samples_per_class=config.training.query_samples_per_class, \
                    transform= transform, sample_frac = 0.1 )

    train_dl = DataLoader(mini_imagenet_dataset_train,batch_size= config.training.batch_size,shuffle= True)
    val_dl = DataLoader(mini_imagenet_dataset_val,batch_size= config.training.batch_size,shuffle= False)

    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(512,config.training.N_way)

    device = 'cpu'
    if config.training.gpu:
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        
    resnet18.to(device)
    optim = torch.optim.Adam(resnet18.parameters(), lr = config.training.lr)
    optim.defaults['inner_lr'] = config.training.lr*10

    iteration = 0

    for epoch in range(config.training.n_epochs):
            
        do_train(iteration , epoch , device,torch.nn.BCEWithLogitsLoss(),optim,AdamExplicitGrad,\
            resnet18, train_dl, val_dl, config)
            
    if config.training.wandb_logging:
        wandb.finish()
        


if __name__ == "__main__":
    main()
