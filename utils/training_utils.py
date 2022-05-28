"""TODO
"""

from tokenize import String
import torch 
import torch.nn as nn
import numpy as np
import os 
import copy
import torchvision.models as models 

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision.models.resnet import ResNet
from typing import *
from tqdm import tqdm

from traitlets import Float

#TODO
def maml_inner_loop_train(loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,\
    model: ResNet, xs: torch.Tensor, ys: torch.Tensor, cfg: Dict):
    """TODO
    """

    model.train()

    for i in range(cfg["inner_loop_steps"]):
        
        optimizer.zero_grad()
        y_pred = model(xs)
        loss = loss_fn(y_pred, torch.eye(cfg["N_way"])[ys].to(ys.device))
        grads = torch.autograd.grad(loss,model.parameters(),create_graph=True)
        named_grads = {}
        for (name , param) , grad in zip(model.named_parameters(),grads):
            named_grads[name] = grad
            setattr(param,"param_name",name)
        state_dict = model.state_dict()
        state_dict.update(optimizer.step(named_grads))
        model.load_state_dict(state_dict)
    

#TODO add type hinting here.
def do_train(device: String, loss_fn: nn.modules.loss._Loss, optimizer: Optimizer,optimizer_inner_loop: Optimizer,\
    model: ResNet, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict)-> torch.float64:
    """Trains MAML Model for one epoch. It also has validations at interval
    of steps

    Parameters
    ----------
    
    device: TODO

    loss_fn: TODO

    optimizer: torch.optim.Optimizer
        One of many torch optimizers like SGD, Adam etc.

    optimizer_inner_loop: TODO

    train_dl: torch.utils.data.DataLoader
        Dataloader for training set. Consists of (xs,ys,xq,yq) 
        where s denotes support set and q denotes query set.
        xs&xq are of shape (batch_sz,support_sz,C,H,W)
    val_dl: torch.utils.data.DataLoader
        Same as train_dl withonly difference of data coming from
        validation set. This is required when one epoch is too huge 
        and you want to validate intermediate steps
    model: torchvision.models.resnet.ResNet
        Model of ResNet family with the fc layer changed to having last
        dimension = N_way
    cfg: Dict
        Config Dictionary storing all information related to N_way,
        K_shot, paths etc.  
        TODO-> change config from dict to a class  

    Returns
    -------

    torch.float64
        Aggregated Validation Loss
        TODO: what aggregation ?
    """

    model.train()
    optimizer.zero_grad()

    #TODO can we make sure if i stop a model inbetween i start from the same
    #iteration

    iter = 0

    for (xs,ys), (xq,yq) in tqdm(train_dl):
        #TODO: spawn multiple processes here
        iter += 1
        query_set_loss = 0.
        accuracy = 0.
        total = 0
        for tasks in range(xs.shape[0]):
            support_set_images = xs[tasks].to(device)
            _support_set_labels = ys[tasks].to(device)
            
            #make the labels from 0 to no_of_unique_labels
            unique_labels, support_set_labels = torch.unique(\
                _support_set_labels, sorted= True, return_inverse=True)
                        
            #have a model copy for each task
            new_model = copy.deepcopy(model)
            new_optimizer = optimizer_inner_loop(new_model.parameters(), lr=optimizer.defaults['lr']) # TODO fix LR

            maml_inner_loop_train(loss_fn, new_optimizer, new_model, support_set_images,\
                support_set_labels, cfg)
            
            query_set_preds = new_model(xq[tasks].to(device))
            _query_set_labels = yq[tasks].to(device)

            #make the labels from 0 to no_of_unique_labels
            _, query_set_labels = torch.unique(\
                _query_set_labels, sorted= True, return_inverse=True)

            query_set_loss += loss_fn(query_set_preds,\
                torch.eye(cfg["N_way"])[query_set_labels].to(query_set_labels.device))

            with torch.no_grad():
                _ , preds = torch.max(query_set_preds.data,1)
                accuracy += (preds == query_set_labels).sum().item()
                total += preds.shape[0]
        query_set_loss/=xs.shape[0]

        #TODO Can we keep a loss global list to update the losses regularly 
        print(f"Loss iteration: {iter} => {query_set_loss} with batch size => {xs.shape[0]}")
        print(f"Accuracy iteration: {iter} => {accuracy*100/total} % with total_preds => {total} ")

        query_set_loss.backward()
        optimizer.step()

        #TODO Implement validation Loop


