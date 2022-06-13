"""Contains utility functions for training MAML

"""
import torchviz
import torch 
import torch.nn as nn
import numpy as np
import os 
import copy
import wandb

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision.models.resnet import ResNet
from typing import *
from tqdm import tqdm
from traitlets import Bool
from torchviz import make_dot

from configs import *
from utils import set_attr, del_attr, load_param_dict, yeild_params

def evaluate_query_set(model: nn.Module , xq_per_task: torch.Tensor, yq_per_task: torch.Tensor, is_train: Bool,\
    loss_fn: nn.modules.loss._Loss) -> Tuple[torch.Tensor, int, int]:
    """Evaluates Query Set Loss, Correct Predictions and Total Predictions.
    When is_train is enabled the process is tracked for backprop
    
    Parameters
    ----------

    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    xq_per_task: torch.Tensor
        Query set images for the task
        shape -> [N_way*query_samples_per_class,C,H,W]
    yq_per_task: torch.Tensor
        Query set labels for the task. Note that the actual labels are mapped from 0 to 
        N_way consistent across support and query set
        shape -> [N_way*query_samples_per_class]
    is_train:Bool
        Decides if meta model is updated or not.
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific   

    Returns
    -------
    Tuple[torch.Tensor, int, int]
        (Query Set Loss, Correct Predictions in Query Set, Total Predicitions in Query Set) 
        Here Query Set is per Task    
    
    """

    with torch.set_grad_enabled(is_train):

        query_set_preds = model(xq_per_task)
        _query_set_labels = yq_per_task

        #make the labels from 0 to no_of_unique_labels
        _, query_set_labels = torch.unique(_query_set_labels, sorted= True,\
            return_inverse=True)

        query_set_loss = loss_fn(query_set_preds, torch.eye(query_set_preds.shape[-1])\
            [query_set_labels].to(query_set_labels.device))

        with torch.no_grad():
            _ , preds = torch.max(query_set_preds.data,1)
            correct_preds = (preds == query_set_labels).sum().item()
            total_preds = preds.shape[0]

    return query_set_loss, correct_preds, total_preds

def clone_model(model:nn.Module ) -> nn.Module:
    """Clones the Model and Sets the name of params in attribute 'param_name'

    Parameters
    ----------
    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way

    Returns
    -------
    nn.Module
        Cloned Model
    """

    new_model = copy.deepcopy(model)
    load_param_dict(new_model , OrderedDict(list(model.named_parameters())))  

    for name,param in new_model.named_parameters():
        param.param_name = name

    return new_model

def update_model(model, grads, param_dict, optimizer):
    """Update the model parameters by gradient descent using the grads

    Parameters
    ----------
    model: nn.Module
        Model of ResNet family with the fc layer changed to having last dimension = N_way 

    grads: Tuple[torch.Tensor, ...]
        Gradients of all the model parameters

    param_dict: dict[str,torch.Tensor]
        Dictionary of all model parameters with key as param name and value as param itself

    optimizer: torch.optim.Optimizer
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates
    
    Returns
    -------
    Tuple[nn.Module, dict[str,torch.Tensor]]
        The new model and updated param dictionary
    """
        
    named_grads = {}

    for (name , _) , grad in zip(param_dict.items(),grads):
        named_grads[name] = grad

    new_param_dict = param_dict
    updated_params = optimizer.step(named_grads)

    new_param_dict.update(updated_params)
    load_param_dict(model,new_param_dict)

    return model, new_param_dict