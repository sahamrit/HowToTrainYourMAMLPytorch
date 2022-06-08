"""This module contains training engine for Model Agnostic Meta Learning paper implementation

This file contains the following functions:
    *maml_inner_loop_train : Responsible for training of each task called fast learning 
        (trying to acquire task specific knowledge)
    *run_episodes: Calls maml_inner_loop_train for a batch of task learning and then 
        updates the meta model (acquires task agnostic knowledge)
    *do_train: Conducts training and validation loop for the entire dataset 
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

def maml_inner_loop_train(loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,\
    model: ResNet, xs: torch.Tensor, ys: torch.Tensor, N_way: int, inner_loop_steps: int) -> None:
    """Implementation of inner training loop of MAML. Responsible for task specific knowledge

    Parameters
    ----------
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific
    optimizer: torch.optim.Optimizer
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates
    model: models.resnet.ResNet
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    xs: torch.Tensor
        Images from the support set of a particular task
        xs.shape -> [N_way*K_shot,C,H,W]
    ys: torch.Tensor
        Labels from the support set of a particular task. Note that the actual labels are mapped from 0 to 
        N_way consistent across support and query set
        ys.shape -> [N_way*K_shot]
    N_way: int
        N in N-way-K-shot learning
    inner_loop_steps: int
        No of training steps for inner loop so that the model adapts sufficient to the
        new task  

    Returns
    -------
    None 
    """

    model.train()

    for i in range(inner_loop_steps):
        
        optimizer.zero_grad()
        y_pred = model(xs)
        loss = loss_fn(y_pred, torch.eye(N_way)[ys].to(ys.device))
        grads = torch.autograd.grad(loss,model.parameters(),create_graph=True)
        named_grads = {}
        for (name , param) , grad in zip(model.named_parameters(),grads):
            named_grads[name] = grad
            setattr(param,"param_name",name)
        state_dict = model.state_dict()
        state_dict.update(optimizer.step(named_grads))
        model.load_state_dict(state_dict)

def run_episodes(is_train:Bool ,loss_fn: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,\
    optimizer_inner_loop: Optimizer,model: ResNet, xs: torch.Tensor, xq: torch.Tensor,\
        ys: torch.Tensor, yq: torch.Tensor, N_way: int, inner_loop_steps: int) -> None: 
    """Run episodes for a batch of tasks which includes inner and outer loop training

    Parameters
    ----------
    is_train:Bool
        Differentiates if meta model is updated or not.
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific
    optimizer: torch.optim.Optimizer
        Any of torch optimizers for the meta model
    optimizer_inner_loop: torch.optim.Optimizer
        Optimizer for the inner loop
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates
    model: models.resnet.ResNet
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    xs: torch.Tensor
        Batch of support set images
        xs.shape -> [batch_size,N_way*K_shot,C,H,W]
    ys: torch.Tensor
        Batch of support set labels. Note that the actual labels are mapped from 0 to 
        N_way consistent across support and query set  
        ys.shape -> [batch_size,N_way*K_shot]
    xq: torch.Tensor
        Batch of query set images
        xs.shape -> [batch_size,N_way*query_samples_per_class,C,H,W]
    yq: torch.Tensor
        Batch of query set labels. Note that the actual labels are mapped from 0 to 
        N_way consistent across support and query set
        ys.shape -> [batch_size,N_way*query_samples_per_class]
    N_way: int
        N in N-way-K-shot learning
    inner_loop_steps: int
        No of training steps for inner loop so that the model adapts sufficient to the
        new task

    Returns
    -------
    None      
    """

    query_set_loss = 0.; correct_preds = 0; total_preds = 0

    for tasks in range(xs.shape[0]):
        support_set_images = xs[tasks]
        _support_set_labels = ys[tasks]
        
        #make the labels from 0 to no_of_unique_labels
        unique_labels, support_set_labels = torch.unique(\
            _support_set_labels, sorted= True, return_inverse=True)
                    
        #have a model copy for each task
        new_model = copy.deepcopy(model)
        new_optimizer = optimizer_inner_loop(new_model.parameters(), lr=optimizer.defaults['lr']) # TODO fix LR

        maml_inner_loop_train(loss_fn, new_optimizer, new_model, support_set_images,\
            support_set_labels,N_way, inner_loop_steps)
        
        with torch.set_grad_enabled(is_train):

            query_set_preds = new_model(xq[tasks])
            _query_set_labels = yq[tasks]

            #make the labels from 0 to no_of_unique_labels
            _, query_set_labels = torch.unique(\
                _query_set_labels, sorted= True, return_inverse=True)

            query_set_loss += loss_fn(query_set_preds,\
                torch.eye(N_way)[query_set_labels].to(query_set_labels.device))

            with torch.no_grad():
                _ , preds = torch.max(query_set_preds.data,1)
                correct_preds += (preds == query_set_labels).sum().item()
                total_preds += preds.shape[0]

    return query_set_loss, correct_preds, total_preds
    
def do_train(iter:int , epoch:int , device: str, loss_fn: nn.modules.loss._Loss, optimizer: Optimizer,optimizer_inner_loop: Optimizer,\
    model: ResNet, train_dl: DataLoader, val_dl: DataLoader, conf: ExperimentConfig )-> None:
    """Trains MAML Model for one epoch. It also has validates the model at regular intervals

    Parameters
    ----------
    iter:int
        global parameter to monitor the total number of iterations
    epoch:int
        global parameter to monitor current epoch
    device: str
        cpu or cuda:0
    loss_fn: nn.modules.loss._Loss
        torch.nn.BCEWithLogitsLoss in specific
    optimizer: torch.optim.Optimizer
        Any of torch optimizers for the meta model
    optimizer_inner_loop: torch.optim.Optimizer
        Optimizer for the inner loop
        engine.optimizers.AdamExplicitGrad which takes specific grad parameter to avoid
        interaction with p.grad. Needed for higher order derivates
    model: models.resnet.ResNet
        Model of ResNet family with the fc layer changed to having last dimension = N_way
    train_dl: torch.utils.data.DataLoader
        Dataloader for training set. Consists of (xs,ys,xq,yq) 
        where s denotes support set and q denotes query set.
        xs&xq are of shape (batch_sz,support_sz,C,H,W)
    val_dl: torch.utils.data.DataLoader
        Same as train_dl with only difference of data coming from
        validation set. This is required when one epoch is too huge 
        and you want to validate intermediate steps
    conf: ExperimentConfig
        centralised config for the experiment

    Returns
    -------
    None
    """

    model.train()
    optimizer.zero_grad()

    #TODO can we make sure if i stop a model inbetween i start from the same
    #iteration


    for (xs,ys), (xq,yq) in tqdm(train_dl,disable=True):
        #TODO: spawn multiple processes here

        xs = xs.to(device); xq = xq.to(device); ys = ys.to(device); yq = yq.to(device)
        iter += 1

        query_set_loss, correct_preds, total_preds = run_episodes(True, loss_fn\
            ,optimizer, optimizer_inner_loop, model, xs, xq\
                ,ys ,yq, train_dl.dataset.N, conf.training.inner_loop_steps)

        if iter == 1:
            make_dot(query_set_loss, params=dict(list(model.named_parameters())), show_saved= True, show_attrs= True )\
                .render(filename = os.path.join(conf.log_dir,conf.curr_run), format = 'png')


        query_set_loss/=xs.shape[0]

        if conf.training.wandb_logging:
            wandb.log({"trainingQuerySetLossPerMinibatch":query_set_loss, "trainingQuerySetAccuracyPerMinibatch":correct_preds*100/total_preds})

        if iter% conf.training.train_verbosity == 0:
            print(f"Training: Epoch: {epoch} Iteration: {iter} Loss => {query_set_loss:.4f} with batch size => {xs.shape[0]}")
            print(f"Training: Epoch: {epoch} Iteration: {iter} Accuracy => {correct_preds*100/total_preds:.2f}% with total_preds => {total_preds} ")

        query_set_loss.backward()
        optimizer.step()

        if iter% conf.training.val_freq == 0:
            model.eval()
            optimizer.zero_grad()

            val_loss = []; correct_preds= 0; total_preds = 0

            for (xs,ys), (xq,yq) in tqdm(val_dl,disable=True):
                #TODO: spawn multiple processes here

                xs = xs.to(device); xq = xq.to(device); ys = ys.to(device); yq = yq.to(device)

                query_set_loss, _correct_preds, _total_preds = run_episodes(False, loss_fn\
                    ,optimizer, optimizer_inner_loop, model, xs, xq\
                        ,ys ,yq, val_dl.dataset.N, conf.training.inner_loop_steps)

                query_set_loss/=xs.shape[0]

                val_loss.append(query_set_loss.cpu()); correct_preds += _correct_preds
                total_preds += _total_preds

            if conf.training.wandb_logging:
                wandb.log({"valLoss":np.mean(val_loss), "valAccuracy":correct_preds*100/total_preds})

            print(f"Validation: Epoch: {epoch} Iteration: {iter} Loss => {np.mean(val_loss):.4f} with batch size => {xs.shape[0]}")
            print(f"Validation: Epoch: {epoch} Iteration: {iter} Accuracy => {correct_preds*100/total_preds:.2f}% with total_preds => {total_preds} ")


            model.train()
            optimizer.zero_grad()

        #TODO Can we keep a loss global list to update the losses regularly 
        #TODO Use Logger
        #TODO Implement ckpting
        #TODO Implement validation Loop
        #TODO where to put model.eval() ??


